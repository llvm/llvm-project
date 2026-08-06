domain: "[N] -> { B[i0] : 0 <= i0 < N; C[]; A[] }"
child:
  schedule: "[N] -> [{ B[i0] -> [(1 + i0)]; C[] -> [(N)]; A[] -> [(0)] }]"
  permutable: 1
  coincident: [ 1 ]
  child:
    sequence:
    - filter: "[N] -> { A[]; B[i0] }"
      child:
        set:
        - filter: "[N] -> { A[] }"
        - filter: "[N] -> { B[i0] }"
    - filter: "[N] -> { C[] }"
