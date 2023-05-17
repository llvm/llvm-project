domain: "{ E[]; G[]; F[]; D[]; B[]; C[]; A[] }"
child:
  sequence:
  - filter: "{ D[]; B[]; C[]; A[] }"
    child:
      set:
      - filter: "{ B[]; C[]; A[] }"
        child:
          sequence:
          - filter: "{ B[]; A[] }"
            child:
              set:
              - filter: "{ A[] }"
              - filter: "{ B[] }"
          - filter: "{ C[] }"
      - filter: "{ D[] }"
  - filter: "{ E[] }"
  - filter: "{ G[]; F[] }"
    child:
      set:
      - filter: "{ F[] }"
      - filter: "{ G[] }"
