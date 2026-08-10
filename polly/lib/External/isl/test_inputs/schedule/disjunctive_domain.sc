# Check that the size computation used for loop coalescing avoidance
# does not get confused by disjunctive domains.
domain: [N] -> { S_9[k, i, j = k] : 0 < k <= -3 + N and k < i < N; S_9[k, k, j] : 0 < k <= -3 + N and k <= j < N; S_9[-2 + N, i, j] : N >= 3 and -2 + N <= i < N and -2 + N <= j < N }
validity: [N] -> { S_9[k, 1 + N, j] -> S_9[1 + k, -1 + N, j'] : 0 < k <= -3 + N and j < N and j' > k and -1 + j <= j' <= j; S_9[-2 + N, i, -2 + N] -> S_9[-2 + N, i, -1 + N] : N >= 3 and -2 + N <= i < N}
