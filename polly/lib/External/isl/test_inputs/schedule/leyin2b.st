domain: "[N] -> { B[i0] : 0 <= i0 < N; C[]; A[] }"
child:
  schedule: "[N] -> [{ B[i0] -> [(i0)]; C[] -> [(N)]; A[] -> [(0)] }]"
  permutable: 1
  coincident: [ 1 ]
  child:
    sequence:
    - filter: "[N] -> { A[] }"
    - filter: "[N] -> { C[] }"
    - filter: "[N] -> { B[i0] }"
