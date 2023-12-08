domain: "[N] -> { S_9[k, i, j = k] : 0 < k <= -3 + N and k < i < N; S_9[k, i = k, j] : 0 < k <= -3 + N and k <= j < N; S_9[k = -2 + N, i, j] : N >= 3 and -2 + N <= i < N and -2 + N <= j < N }"
child:
  schedule: "[N] -> [{ S_9[k, i, j] -> [(k)] }, { S_9[k, i, j] -> [(2k + i)] }, { S_9[k, i, j] -> [(k + j)] }]"
  permutable: 1
  coincident: [ 1, 1, 1 ]
