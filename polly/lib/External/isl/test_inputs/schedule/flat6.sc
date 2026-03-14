# Check that a fixed value of one dimension in terms of the others
# does not cause loop coalescing avoidance to break down.
domain: { S[a, b] : -b >= 0 and a + 10b >= 0 and -a - b + 9 >= 0 }
