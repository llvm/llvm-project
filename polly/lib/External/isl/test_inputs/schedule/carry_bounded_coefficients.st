domain: "{ C[i0, i1] : 2 <= i0 <= 3999 and 0 <= i1 < i0 }"
child:
  schedule: "[{ C[i0, i1] -> [(i0)] }]"
  child:
    schedule: "[{ C[i0, i1] -> [(i1)] }]"
