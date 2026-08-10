# Earlier versions of isl would fail to produce an AST for this input
# due to a simplification bug.
domain: "{ S_3[i, j, k] : 0 <= i <= 12 and 0 <= j < i and 0 <= k < j }"
child:
  schedule: "[{ S_3[i, j, k] -> [(3993j + 1997k)] }]"
  child:
    schedule: "[{ S_3[i, j, k] -> [(32*floor((2j + k)/32))] }, { S_3[i, j, k] -> [(32*floor((i)/32))] }]"
    child:
      schedule: "[{ S_3[i, j, k] -> [(2j + k - 32*floor((2j + k)/32))] }, { S_3[i, j, k] -> [(i)] }]"
