domain: "{ B[i0] : 0 <= i0 <= 31; C[]; A[] }"
child:
  schedule: "[{ B[i0] -> [(i0)]; C[] -> [(31)]; A[] -> [(0)] }]"
  permutable: 1
  coincident: [ 1 ]
  child:
    sequence:
    - filter: "{ A[] }"
    - filter: "{ B[i0] }"
    - filter: "{ C[] }"
