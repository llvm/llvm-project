# Check that a fixed value of one dimension in terms of the others
# does not cause loop coalescing avoidance to break down.
domain: { S[a, floor(a/2)] : 0 <= a <= 9 }
