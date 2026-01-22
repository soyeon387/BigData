import json
import heapq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 1) 데이터 로딩

def load_segments_csv(path: str) -> pd.DataFrame:
    """
    서울열린데이터광장 '서울교통공사_역간거리' CSV를 읽어서
    같은 호선 내 인접역 간선(u->v) 테이블을 만든다.
    기대 컬럼(예상): 연번, 호선, 역명, 운행시간(분), 역간거리(km)
    """
    df = pd.read_csv(path, encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]

    col_line = "호선"
    col_station = "역명"
    col_seq = "연번"
    col_time = "운행시간(분)"
    col_dist = "역간거리(km)"

    for c in [col_line, col_station, col_seq, col_time, col_dist]:
        if c not in df.columns:
            raise ValueError(f"[segments.csv] 컬럼 '{c}'를 찾지 못했습니다. 실제 컬럼: {list(df.columns)}")

    df = df.sort_values([col_line, col_seq]).reset_index(drop=True)
    df["next_station"] = df.groupby(col_line)[col_station].shift(-1)

    edges = df.dropna(subset=["next_station"]).copy()
    edges = edges.rename(columns={
        col_line: "line",
        col_station: "u_station",
        "next_station": "v_station",
        col_time: "time_min",
        col_dist: "dist_km",
    })[["line", "u_station", "v_station", "time_min", "dist_km"]]

    # 양방향 간선
    rev = edges.rename(columns={"u_station": "v_station", "v_station": "u_station"})
    edges = pd.concat([edges, rev], ignore_index=True)

    # 숫자형 변환
    edges["time_min"] = pd.to_numeric(edges["time_min"], errors="coerce").fillna(1.0)
    edges["dist_km"] = pd.to_numeric(edges["dist_km"], errors="coerce").fillna(0.1)

    return edges


def load_transfers_json(path: str) -> pd.DataFrame:
    """
    공공데이터포털 '환승역거리 소요시간' JSON을 읽는다.
    컬럼명은 버전에 따라 다를 수 있어, 아래 key 후보로 최대한 매칭한다.
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, dict):
        list_candidates = [v for v in obj.values() if isinstance(v, list)]
        if not list_candidates:
            raise ValueError("transfers.json에서 리스트 형태 데이터를 찾지 못했습니다.")
        rows = list_candidates[0]
    else:
        rows = obj

    df = pd.DataFrame(rows)
    df.columns = [c.strip() for c in df.columns]

    # 환승역명 / 환승거리 / 환승소요시간(초) 컬럼 후보들
    station_candidates = ["환승역", "환승역명", "STATION_NM", "STATN_NM", "역명"]
    dist_candidates = ["환승거리(m)", "환승거리", "TRANSFER_DIST", "DIST_M", "거리(m)"]
    time_candidates = ["환승소요시간(초)", "환승소요시간", "TRANSFER_TIME", "TIME_SEC", "소요시간(초)"]

    def pick_col(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    col_station = pick_col(station_candidates)
    col_dist = pick_col(dist_candidates)
    col_time = pick_col(time_candidates)

    if not col_station or not col_dist or not col_time:
        raise ValueError(
            f"[transfers.json] 필요한 컬럼을 자동으로 못 찾았습니다.\n"
            f"현재 컬럼: {list(df.columns)}\n"
            f"찾은 값: station={col_station}, dist={col_dist}, time={col_time}"
        )

    out = df.rename(columns={
        col_station: "station",
        col_dist: "transfer_dist_m",
        col_time: "transfer_time_sec",
    })[["station", "transfer_dist_m", "transfer_time_sec"]].copy()

    out["transfer_dist_m"] = pd.to_numeric(out["transfer_dist_m"], errors="coerce").fillna(200.0)
    out["transfer_time_sec"] = pd.to_numeric(out["transfer_time_sec"], errors="coerce").fillna(180.0)

    return out


def load_facilities_csv(path: str) -> pd.DataFrame:
    """
    (선택) 역별 시설 유무 CSV.
    최소 컬럼 예시: station, has_elevator, has_escalator
    """
    df = pd.read_csv(path, encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]

    need = ["station", "has_elevator", "has_escalator"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"[facilities.csv] 컬럼 '{c}' 필요. 현재: {list(df.columns)}")

    df["has_elevator"] = pd.to_numeric(df["has_elevator"], errors="coerce").fillna(0).astype(int)
    df["has_escalator"] = pd.to_numeric(df["has_escalator"], errors="coerce").fillna(0).astype(int)
    return df


# 2) 환승 난이도 점수
def minmax(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mn) / (mx - mn)


def compute_transfer_difficulty(transfers: pd.DataFrame, facilities: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    환승역별 난이도(0~100) 계산.
    - 거리/시간이 길수록 ↑
    - 엘리베이터/에스컬레이터 있으면 ↓
    """
    df = transfers.copy()

    # 정규화
    df["dist_n"] = minmax(df["transfer_dist_m"])
    df["time_n"] = minmax(df["transfer_time_sec"])

    # 기본 점수
    df["score_base"] = (0.45 * df["dist_n"] + 0.55 * df["time_n"]) * 100.0

    if facilities is not None:
        df = df.merge(facilities, on="station", how="left")
        df["has_elevator"] = df["has_elevator"].fillna(0)
        df["has_escalator"] = df["has_escalator"].fillna(0)

        # 시설이 있으면 감점(난이도 감소)
        df["score"] = df["score_base"] - (df["has_elevator"] * 25) - (df["has_escalator"] * 10)
    else:
        df["score"] = df["score_base"]

    df["score"] = df["score"].clip(0, 100)

    return df[["station", "transfer_dist_m", "transfer_time_sec", "score"]]


# 3) 그래프 만들기 + 최단경로(피로도 최소)
def build_graph(segment_edges: pd.DataFrame, transfer_scores: pd.DataFrame, alpha: float = 0.03):
    """
    노드: '호선|역명' 형태로 분리(같은 역명이 다른 호선에 있을 수 있어서)
    간선 가중치:
      - 역간 이동: time_min (분)
      - 환승: (transfer_time_sec/60) + alpha*score
    alpha: 난이도 점수(0~100)를 몇 "분"으로 환산할지
    """
    graph = {}

    def add_edge(a, b, w):
        graph.setdefault(a, []).append((b, float(w)))

    # 1) 호선 내 이동 간선
    for _, r in segment_edges.iterrows():
        u = f"{r['line']}|{r['u_station']}"
        v = f"{r['line']}|{r['v_station']}"
        add_edge(u, v, r["time_min"])

    # 2) 환승 간선: 같은 역명이 여러 호선에 있으면 서로 연결
    station_to_lines = segment_edges.groupby("u_station")["line"].apply(lambda x: sorted(set(x))).to_dict()

    transfer_map = transfer_scores.set_index("station").to_dict("index")

    for station, lines in station_to_lines.items():
        if len(lines) < 2:
            continue
        if station not in transfer_map:
            continue

        t_sec = transfer_map[station]["transfer_time_sec"]
        score = transfer_map[station]["score"]
        w_transfer = (t_sec / 60.0) + alpha * score

        # 모든 호선 조합 연결(완전그래프)
        nodes = [f"{line}|{station}" for line in lines]
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                add_edge(nodes[i], nodes[j], w_transfer)
                add_edge(nodes[j], nodes[i], w_transfer)

    return graph


def dijkstra(graph, start, goal):
    pq = [(0.0, start)]
    dist = {start: 0.0}
    prev = {start: None}

    while pq:
        cur_d, u = heapq.heappop(pq)
        if u == goal:
            break
        if cur_d != dist.get(u, np.inf):
            continue

        for v, w in graph.get(u, []):
            nd = cur_d + w
            if nd < dist.get(v, np.inf):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if goal not in dist:
        return None, np.inf

    # 경로 복원
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path, dist[goal]


def pick_start_goal_nodes(segment_edges: pd.DataFrame, start_station: str, end_station: str):
    """
    같은 역명이 여러 호선에 있으면 '가장 유리한 시작/도착 노드'를 고르기 위해
    모든 후보를 만들고, 나중에 전체 조합에서 최솟값을 찾는다.
    """
    station_lines = segment_edges.groupby("u_station")["line"].apply(lambda x: sorted(set(x))).to_dict()
    s_lines = station_lines.get(start_station, [])
    e_lines = station_lines.get(end_station, [])

    if not s_lines or not e_lines:
        raise ValueError("출발역/도착역이 segments.csv에 없습니다. 역명 철자(띄어쓰기) 확인해줘.")

    start_nodes = [f"{ln}|{start_station}" for ln in s_lines]
    end_nodes = [f"{ln}|{end_station}" for ln in e_lines]
    return start_nodes, end_nodes


def recommend_route(graph, segment_edges, start_station: str, end_station: str):
    start_nodes, end_nodes = pick_start_goal_nodes(segment_edges, start_station, end_station)

    best = (None, np.inf)
    for s in start_nodes:
        for g in end_nodes:
            path, cost = dijkstra(graph, s, g)
            if cost < best[1]:
                best = (path, cost)

    return best  # (path, total_cost)


# 4) 시각화(난이도 TOP)
def plot_top_hard_transfers(transfer_scores: pd.DataFrame, top_n: int = 20):
    df = transfer_scores.sort_values("score", ascending=False).head(top_n)
    plt.figure()
    plt.barh(df["station"][::-1], df["score"][::-1])
    plt.xlabel("Transfer Difficulty (0~100)")
    plt.title(f"Top {top_n} Hardest Transfer Stations")
    plt.tight_layout()
    plt.show()


# 5) 실행 예시
if __name__ == "__main__":
    # 파일 경로
    SEG_PATH = "data/segments.csv"
    TRN_PATH = "data/transfers.json"
    FAC_PATH = "data/facilities.csv"  # 없으면 None 

    # 1) 로딩
    segments = load_segments_csv(SEG_PATH)
    transfers = load_transfers_json(TRN_PATH)

    try:
        facilities = load_facilities_csv(FAC_PATH)
    except FileNotFoundError:
        facilities = None

    # 2) 환승 난이도 계산
    transfer_scores = compute_transfer_difficulty(transfers, facilities)

    # 3) 그래프 생성 
    graph = build_graph(segments, transfer_scores, alpha=0.03)

    # 4) 난이도 TOP 시각화
    plot_top_hard_transfers(transfer_scores, top_n=15)

    # 5) 경로 추천 테스트
    start = "서울역"
    end = "강남"
    path, cost = recommend_route(graph, segments, start, end)

    if path is None:
        print("경로를 찾지 못했습니다.")
    else:
        print(f"\n[추천 경로] {start} -> {end}")
        print(" -> ".join(path))
        print(f"총 비용(분 단위, 시간+환승피로 반영): {cost:.1f}")
