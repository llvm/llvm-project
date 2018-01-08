domain: "{ B[0]; C[]; A[] }"
child:
  set:
  - filter: "{ B[i0]; A[] }"
    child:
      schedule: "[{ B[i0] -> [(1)]; A[] -> [(0)] }]"
      child:
        set:
        - filter: "{ A[] }"
        - filter: "{ B[i0] }"
  - filter: "{ C[] }"
