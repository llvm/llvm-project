domain: "{ B[]; C[]; A[] }"
child:
  sequence:
  - filter: "{ B[]; A[] }"
    child:
      set:
      - filter: "{ A[] }"
      - filter: "{ B[] }"
  - filter: "{ C[] }"
