# Check that a fixed value of one dimension in terms of the others
# does not cause loop coalescing avoidance to break down.
domain: { S[0, a, floor(a/2), floor(a/4)] : 0 <= a <= 9 }
