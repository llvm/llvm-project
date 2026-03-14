domain: "[n] -> { A[i, j, k] : 0 <= i < n and 0 <= j < n and 0 <= k < n; B[i, j, k] : 0 <= i < n and 0 <= j < n and 0 <= k < n }"
child:
  sequence:
  - filter: "[n] -> { A[i, j, k] }"
    child:
      schedule: "[n] -> [{ A[i, j, k] -> [(i)] }, { A[i, j, k] -> [(j)] }, { A[i, j, k] -> [(k)] }]"
      permutable: 1
      coincident: [ 1, 1, 0 ]
  - filter: "[n] -> { B[i, j, k] }"
    child:
      schedule: "[n] -> [{ B[i, j, k] -> [(i)] }, { B[i, j, k] -> [(j)] }, { B[i, j, k] -> [(k)] }]"
      permutable: 1
      coincident: [ 1, 1, 0 ]
