# Check that a fixed value of one dimension in terms of the others
# does not cause loop coalescing avoidance to break down.
domain: [M] -> { S[a, b] : 0 <= a <= 99 and M <= 2b <= 1 + M }
