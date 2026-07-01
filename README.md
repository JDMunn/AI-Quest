This was written entirely on Anthropic's Sonnet 5. I wrote this as a refresher tool to help me learn the concepts behind popular AI algorithm types. 
This the README written by the creator. 

# AI Quest — Module 1: K-Means Quest

A terminal game for learning K-means clustering by actually doing the math,
not just reading about it.

## Run it

```
python3 kmeans_quest.py
```

That's it — no `pip install` needed. It only uses the Python standard
library. If you happen to have `matplotlib` installed, you'll also get a
bonus scatter-plot image saved at the end, but it's entirely optional.

Requires Python 3.7+.

## What it teaches

- The K-means objective function (what it's actually trying to minimize)
- The assignment step, with the full Euclidean distance formula worked out
  for every point
- The update step, with the centroid (mean) formula worked out for every
  cluster
- Why initialization matters (you can pick your own starting centroids and
  watch a bad choice lead to a worse final answer)
- Convergence, and reading the within-cluster-sum-of-squares (WCSS) score

## How it plays

1. You're shown a set of unlabeled 2D points.
2. You choose how to initialize K starting centroids — randomly, or by
   picking existing points yourself.
3. Each round, the game quizzes you: "which centroid is this point closest
   to?" and "what's the new centroid for this cluster?" — then reveals the
   full worked-out math either way, and scores you.
4. This repeats until the algorithm converges, then you get a final score
   and rank (Clustering Rookie → K-Means Apprentice → Clustering Master).

Three levels are included, from a simple 2-cluster warm-up to a harder
3-cluster challenge with overlapping data.

## What's next

This is Module 1 of a planned series (AI Quest) covering other ways to
build AI models — decision trees, neural networks, NLP, and more — using
the same "play it, don't just read it" approach. Let me know which one
you'd like built next.
