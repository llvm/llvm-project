# This is a test case with an expansion node.
# It requires some coalescing to avoid splitting up the inner loop
# in the main loop nest.
domain: "[n] -> { G[i] : (1 + i) mod 2 = 0 and 0 <= i < n; G[i = -1 + n] : n > 0 }"
child:
  schedule: "[{ G[i] -> [(i)] }]"
  child:
    contraction: "[n] -> { S[i] -> G[(-1 + n)] : 2*floor((i)/2) >= -2 + n; S[i] -> G[(1 + i - (i) mod 2)] : i <= -3 + n or (i = -2 + n and (1 + n) mod 2 = 0) }"
    expansion: "[n] -> { G[j] -> S[i] : (1 + j) mod 2 = 0 and -1 + j <= i <= j and i <= -3 + n; G[j = -1 + n] -> S[i] : i < n and 2*floor((i)/2) >= -2 + n; G[j = -2 + n] -> S[i = -2 + n] : (1 + n) mod 2 = 0 }"
