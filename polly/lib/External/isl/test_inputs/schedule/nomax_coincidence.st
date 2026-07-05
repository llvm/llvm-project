domain: "[n] -> { A[i, j, k] : 0 <= i < n and 0 <= j < n and 0 <= k < n; B[i, j, k] : 0 <= i < n and 0 <= j < n and 0 <= k < n }"
child:
  schedule: "[n] -> [{ A[i, j, k] -> [(i)]; B[i, j, k] -> [(i)] }, { A[i, j, k] -> [(j)]; B[i, j, k] -> [(k)] }, { A[i, j, k] -> [(k)]; B[i, j, k] -> [(j)] }]"
  permutable: 1
  coincident: [ 1, 0, 0 ]
  child:
    sequence:
    - filter: "[n] -> { A[i, j, k] }"
    - filter: "[n] -> { B[i, j, k] }"
