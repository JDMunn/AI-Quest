#!/usr/bin/env python3
"""
K-MEANS QUEST
=============
Module 1 of "AI Quest" — a series of terminal games for learning the math
and concepts behind AI/ML methods by actually playing with them.

Run it with:  python3 kmeans_quest.py

No third-party packages required. If matplotlib is installed, you'll also
get an optional pop-up plot at the end — but the game works with just the
Python standard library.
"""

import math
import os
import random
import sys
import time

# ----------------------------------------------------------------------
# Terminal setup (enables ANSI colors on Windows terminals too)
# ----------------------------------------------------------------------
if os.name == "nt":
    os.system("")

USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def c(text, code):
    """Wrap text in an ANSI color code, if colors are enabled."""
    if not USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


BOLD = "1"
DIM = "2"
CYAN = "36"
MAGENTA = "35"
YELLOW = "33"
GREEN = "32"
RED = "31"
BLUE = "34"

CLUSTER_COLORS = [CYAN, MAGENTA, YELLOW, GREEN, BLUE, RED]


def pause(seconds=0.6):
    time.sleep(seconds)


def header(title):
    line = "=" * (len(title) + 4)
    print("\n" + c(line, BOLD))
    print(c(f"  {title}", BOLD))
    print(c(line, BOLD))


def divider():
    print(c("-" * 60, DIM))


# ----------------------------------------------------------------------
# Core K-means math (pure Python, mirrors the formulas we teach)
# ----------------------------------------------------------------------

def dist(p, q):
    """Euclidean distance: d(p, q) = sqrt(sum((p_i - q_i)^2))"""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p, q)))


def assign_clusters(points, centroids):
    assignments = []
    for p in points:
        distances = [dist(p, cen) for cen in centroids]
        assignments.append(distances.index(min(distances)))
    return assignments


def compute_centroids(points, assignments, k, old_centroids):
    sums = [[0.0, 0.0] for _ in range(k)]
    counts = [0] * k
    for p, a in zip(points, assignments):
        sums[a][0] += p[0]
        sums[a][1] += p[1]
        counts[a] += 1
    new_centroids = []
    for i in range(k):
        if counts[i] == 0:
            new_centroids.append(old_centroids[i])  # empty cluster: keep old spot
        else:
            new_centroids.append((sums[i][0] / counts[i], sums[i][1] / counts[i]))
    return new_centroids


def wcss(points, assignments, centroids):
    """Within-cluster sum of squares (the objective K-means minimizes)."""
    return sum(dist(p, centroids[a]) ** 2 for p, a in zip(points, assignments))


# ----------------------------------------------------------------------
# Dataset generation
# ----------------------------------------------------------------------

def make_blobs(centers, points_per_blob, spread, seed=None):
    rng = random.Random(seed)
    points = []
    for cx, cy in centers:
        for _ in range(points_per_blob):
            x = round(rng.gauss(cx, spread), 2)
            y = round(rng.gauss(cy, spread), 2)
            points.append((x, y))
    rng.shuffle(points)
    return points


LEVELS = {
    "1": {
        "name": "Level 1: Two Neighborhoods",
        "k": 2,
        "centers": [(2, 2), (10, 10)],
        "points_per_blob": 4,
        "spread": 1.0,
    },
    "2": {
        "name": "Level 2: Three Districts",
        "k": 3,
        "centers": [(1, 1), (10, 2), (5, 10)],
        "points_per_blob": 4,
        "spread": 1.2,
    },
    "3": {
        "name": "Level 3: Challenge — Overlapping Zones",
        "k": 3,
        "centers": [(3, 3), (6, 5), (9, 3)],
        "points_per_blob": 5,
        "spread": 1.8,
    },
}


# ----------------------------------------------------------------------
# ASCII visualization
# ----------------------------------------------------------------------

CLUSTER_SYMBOLS = "123456"
CENTROID_LETTERS = "ABCDEF"


def ascii_plot(points, assignments=None, centroids=None, width=58, height=20):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    if centroids:
        xs += [cen[0] for cen in centroids]
        ys += [cen[1] for cen in centroids]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    pad_x = (max_x - min_x) * 0.15 or 1.0
    pad_y = (max_y - min_y) * 0.15 or 1.0
    min_x -= pad_x
    max_x += pad_x
    min_y -= pad_y
    max_y += pad_y

    grid = [[" " for _ in range(width)] for _ in range(height)]

    def to_cell(x, y):
        col = int((x - min_x) / (max_x - min_x) * (width - 1))
        row = int((height - 1) - (y - min_y) / (max_y - min_y) * (height - 1))
        col = min(max(col, 0), width - 1)
        row = min(max(row, 0), height - 1)
        return row, col

    for i, p in enumerate(points):
        row, col = to_cell(*p)
        if assignments is not None:
            sym = CLUSTER_SYMBOLS[assignments[i] % len(CLUSTER_SYMBOLS)]
        else:
            sym = "."
        grid[row][col] = sym

    if centroids:
        for i, cen in enumerate(centroids):
            row, col = to_cell(*cen)
            grid[row][col] = CENTROID_LETTERS[i % len(CENTROID_LETTERS)]

    print(c("+" + "-" * width + "+", DIM))
    for row in grid:
        line = "".join(row)
        if USE_COLOR:
            colored = ""
            for ch in line:
                if ch in CLUSTER_SYMBOLS:
                    idx = CLUSTER_SYMBOLS.index(ch)
                    colored += c(ch, CLUSTER_COLORS[idx % len(CLUSTER_COLORS)])
                elif ch in CENTROID_LETTERS:
                    colored += c(ch, BOLD)
                else:
                    colored += ch
            line = colored
        print(c("|", DIM) + line + c("|", DIM))
    print(c("+" + "-" * width + "+", DIM))


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------

class Score:
    def __init__(self):
        self.points = 0
        self.max_points = 0
        self.correct = 0
        self.total = 0

    def award(self, is_correct, value):
        self.max_points += value
        self.total += 1
        if is_correct:
            self.points += value
            self.correct += 1
            return value
        return 0

    def rank(self):
        if self.max_points == 0:
            return "Unranked"
        pct = self.points / self.max_points
        if pct >= 0.85:
            return "Clustering Master"
        if pct >= 0.6:
            return "K-Means Apprentice"
        return "Clustering Rookie"


# ----------------------------------------------------------------------
# Input helpers
# ----------------------------------------------------------------------

def ask_choice(prompt, valid_options):
    while True:
        ans = input(prompt).strip().upper()
        if ans in valid_options:
            return ans
        print(c(f"   Please enter one of: {', '.join(valid_options)}", RED))


def ask_point(prompt):
    while True:
        raw = input(prompt).strip().replace("(", "").replace(")", "")
        try:
            x_str, y_str = raw.split(",")
            return (float(x_str), float(y_str))
        except ValueError:
            print(c("   Please answer in the format x,y  (e.g. 3.5,2)", RED))


# ----------------------------------------------------------------------
# Teaching content
# ----------------------------------------------------------------------

def show_intro():
    header("K-MEANS QUEST")
    print("\nWelcome! Your quest: sort a scattered set of points into K groups,")
    print("the same way K-means clustering does it — by hand, with the real math.")
    pause()
    divider()
    print(c("\nThe idea:", BOLD))
    print("K-means groups points so that points in the same cluster are as")
    print("close as possible to their cluster's center (the 'centroid'), and")
    print("as far as possible from other clusters' centers.")
    pause()
    print(c("\nWhat it's minimizing (the objective function):", BOLD))
    print(c("   J = Σ_{i=1}^{k}  Σ_{x ∈ C_i}  || x - μ_i ||²", CYAN))
    print("   (total squared distance from every point to its own cluster's centroid)")
    pause()
    print(c("\nThe algorithm, in 4 steps:", BOLD))
    print("   1. Choose K, and pick K starting centroids.")
    print("   2. " + c("Assign:", GREEN) + " put each point in the cluster of its nearest centroid.")
    print(c("        d(x, μ) = √( Σ_j (x_j - μ_j)² )", CYAN))
    print("   3. " + c("Update:", GREEN) + " move each centroid to the mean of its assigned points.")
    print(c("        μ_i = (1/|C_i|) Σ_{x ∈ C_i} x", CYAN))
    print("   4. Repeat steps 2-3 until the assignments stop changing.")
    pause()
    divider()
    input(c("\nPress Enter to pick a level...", DIM))


def choose_level():
    header("CHOOSE YOUR LEVEL")
    for key, lvl in LEVELS.items():
        print(f"  {key}) {lvl['name']}  (K={lvl['k']})")
    choice = ask_choice("\nLevel: ", list(LEVELS.keys()))
    return LEVELS[choice]


def label_points(points):
    return [f"P{i+1}" for i in range(len(points))]


def show_dataset(points, labels):
    header("YOUR DATA")
    print("Here are the unlabeled points you need to cluster:\n")
    for lbl, p in zip(labels, points):
        print(f"   {lbl}: ({p[0]:.2f}, {p[1]:.2f})")
    print()
    ascii_plot(points)


def choose_initial_centroids(points, labels, k):
    header("INITIALIZE CENTROIDS")
    print("Bad starting centroids can lead K-means to a worse final answer —")
    print("this is one of the real weaknesses of the algorithm.\n")
    print("  R) Place K centroids randomly")
    print("  P) Pick K existing data points to use as starting centroids")
    mode = ask_choice("\nChoice: ", ["R", "P"])

    if mode == "R":
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        rng = random.Random()
        centroids = [
            (round(rng.uniform(min(xs), max(xs)), 2), round(rng.uniform(min(ys), max(ys)), 2))
            for _ in range(k)
        ]
    else:
        centroids = []
        chosen = set()
        for i in range(k):
            while True:
                raw = input(f"  Pick a point for centroid {CENTROID_LETTERS[i]} (e.g. P3): ").strip().upper()
                if raw in labels and raw not in chosen:
                    chosen.add(raw)
                    centroids.append(points[labels.index(raw)])
                    break
                print(c("   Enter a point label you haven't used yet, e.g. P1", RED))

    print("\nStarting centroids:")
    for i, cen in enumerate(centroids):
        print(f"   {CENTROID_LETTERS[i]}: ({cen[0]:.2f}, {cen[1]:.2f})")
    ascii_plot(points, centroids=centroids)
    return centroids


def quiz_assignment(point, label, centroids, score):
    print(f"\n{c('Quiz:', YELLOW)} Which centroid is {label} {point} closest to?")
    for i, cen in enumerate(centroids):
        print(f"   {CENTROID_LETTERS[i]}) ({cen[0]:.2f}, {cen[1]:.2f})")
    guess = ask_choice("   Your answer: ", [CENTROID_LETTERS[i] for i in range(len(centroids))])

    distances = [dist(point, cen) for cen in centroids]
    correct_idx = distances.index(min(distances))
    correct_letter = CENTROID_LETTERS[correct_idx]

    print("\n   Checking the math:")
    for i, cen in enumerate(centroids):
        dx = point[0] - cen[0]
        dy = point[1] - cen[1]
        print(
            f"   d({label}, {CENTROID_LETTERS[i]}) = "
            f"√(({point[0]:.2f}-{cen[0]:.2f})² + ({point[1]:.2f}-{cen[1]:.2f})²) "
            f"= √({dx**2:.2f} + {dy**2:.2f}) ≈ {distances[i]:.2f}"
        )

    is_correct = guess == correct_letter
    earned = score.award(is_correct, 10)
    if is_correct:
        print(c(f"   Correct! +{earned} points", GREEN))
    else:
        print(c(f"   Not quite — {label} is actually closest to {correct_letter}.", RED))
    return correct_idx


def show_assignment_line(point, label, centroids, correct_idx):
    distances = [dist(point, cen) for cen in centroids]
    dist_str = ", ".join(f"{CENTROID_LETTERS[i]}={d:.2f}" for i, d in enumerate(distances))
    print(f"   {label} {point} -> distances: {dist_str} -> assigned to {CENTROID_LETTERS[correct_idx]}")


def quiz_centroid_update(cluster_points, cluster_label, score):
    print(f"\n{c('Quiz:', YELLOW)} Cluster {cluster_label} has these points: {cluster_points}")
    print("   Formula: μ = (1/n) Σ x_i")
    guess = ask_point("   New centroid (format x,y): ")

    n = len(cluster_points)
    sum_x = sum(p[0] for p in cluster_points)
    sum_y = sum(p[1] for p in cluster_points)
    correct = (sum_x / n, sum_y / n)

    is_correct = abs(guess[0] - correct[0]) < 0.2 and abs(guess[1] - correct[1]) < 0.2

    x_terms = " + ".join(f"{p[0]:.2f}" for p in cluster_points)
    y_terms = " + ".join(f"{p[1]:.2f}" for p in cluster_points)
    print(f"\n   μ_{cluster_label} = (({x_terms})/{n}, ({y_terms})/{n})")
    print(f"   μ_{cluster_label} = ({sum_x:.2f}/{n}, {sum_y:.2f}/{n}) = ({correct[0]:.2f}, {correct[1]:.2f})")

    earned = score.award(is_correct, 15)
    if is_correct:
        print(c(f"   Correct! +{earned} points", GREEN))
    else:
        print(c(f"   Not quite — the real answer is ({correct[0]:.2f}, {correct[1]:.2f})", RED))
    return correct


def show_update_line(cluster_points, cluster_label, correct):
    n = len(cluster_points)
    print(f"   μ_{cluster_label}: mean of {n} points = ({correct[0]:.2f}, {correct[1]:.2f})")


def run_level(level):
    header(level["name"])
    k = level["k"]
    points = make_blobs(level["centers"], level["points_per_blob"], level["spread"])
    labels = label_points(points)
    score = Score()

    show_dataset(points, labels)
    centroids = choose_initial_centroids(points, labels, k)

    prev_assignments = None
    max_iters = 8
    quiz_budget_per_round = min(3, len(points))

    for iteration in range(1, max_iters + 1):
        header(f"ITERATION {iteration} — ASSIGN STEP")
        distances_all = [[dist(p, cen) for cen in centroids] for p in points]
        assignments = [d.index(min(d)) for d in distances_all]

        quiz_indices = set(random.sample(range(len(points)), quiz_budget_per_round))
        for i, (p, lbl) in enumerate(zip(points, labels)):
            if i in quiz_indices:
                quiz_assignment(p, lbl, centroids, score)
            else:
                show_assignment_line(p, lbl, centroids, assignments[i])

        ascii_plot(points, assignments, centroids)

        header(f"ITERATION {iteration} — UPDATE STEP")
        clusters = [[] for _ in range(k)]
        for p, a in zip(points, assignments):
            clusters[a].append(p)

        quiz_cluster = (iteration - 1) % k
        new_centroids = list(centroids)
        for ci in range(k):
            label_letter = CENTROID_LETTERS[ci]
            if not clusters[ci]:
                print(f"\n   Cluster {label_letter} has no points assigned — centroid stays put.")
                new_centroids[ci] = centroids[ci]
                continue
            if ci == quiz_cluster:
                new_centroids[ci] = quiz_centroid_update(clusters[ci], label_letter, score)
            else:
                n = len(clusters[ci])
                mx = sum(p[0] for p in clusters[ci]) / n
                my = sum(p[1] for p in clusters[ci]) / n
                new_centroids[ci] = (mx, my)
                show_update_line(clusters[ci], label_letter, new_centroids[ci])

        current_wcss = wcss(points, assignments, new_centroids)
        print(f"\n   Current WCSS (J) = {current_wcss:.2f}   (lower = tighter clusters)")

        if assignments == prev_assignments:
            print(c("\nAssignments didn't change — K-means has converged!", GREEN + ";" + BOLD))
            centroids = new_centroids
            break

        prev_assignments = assignments
        centroids = new_centroids
        pause(0.4)

    header("FINAL RESULT")
    final_assignments = assign_clusters(points, centroids)
    ascii_plot(points, final_assignments, centroids)
    final_wcss = wcss(points, final_assignments, centroids)
    print(f"\nFinal WCSS (J) = Σ ||x - μ||² = {final_wcss:.2f}")
    print("(This is the number K-means is trying to minimize. Try re-running with")
    print(" different starting centroids — you may land on a different, sometimes")
    print(" worse, final clustering. That sensitivity to initialization is a real")
    print(" limitation of K-means!)")

    header("YOUR SCORE")
    print(f"Correct predictions: {score.correct}/{score.total}")
    print(f"Points: {score.points}/{score.max_points}")
    print(f"Rank: {c(score.rank(), BOLD)}")

    maybe_matplotlib_plot(points, final_assignments, centroids, level["name"])


def maybe_matplotlib_plot(points, assignments, centroids, title):
    try:
        import matplotlib
        matplotlib.use("Agg")  # safe default; switch below if a display exists
        import matplotlib.pyplot as plt
    except ImportError:
        return

    try:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        colors = [assignments[i] for i in range(len(points))]
        plt.figure(figsize=(6, 5))
        plt.scatter(xs, ys, c=colors, cmap="tab10", s=80, edgecolors="black")
        cxs = [cen[0] for cen in centroids]
        cys = [cen[1] for cen in centroids]
        plt.scatter(cxs, cys, c="black", marker="X", s=200, label="Centroids")
        plt.title(title)
        plt.legend()
        out_path = os.path.join(os.getcwd(), "kmeans_result.png")
        plt.savefig(out_path)
        print(f"\n(Bonus: a matplotlib plot was also saved to {out_path})")
    except Exception:
        pass


def main():
    try:
        show_intro()
        while True:
            level = choose_level()
            run_level(level)
            again = ask_choice("\nPlay another level? (Y/N): ", ["Y", "N"])
            if again == "N":
                break
        header("THANKS FOR PLAYING")
        print("Coming up next in AI Quest: Decision Trees, Neural Networks, and NLP.")
        print("See you on the next quest!\n")
    except KeyboardInterrupt:
        print("\n\nQuest paused. Run the script again anytime to continue learning!")


if __name__ == "__main__":
    main()
