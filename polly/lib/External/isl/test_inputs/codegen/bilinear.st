# Check that the generated code does not contain a loop upper bound
# that depends on the loop iterator.
domain: "[d, w] -> { A[i0] : i0 >= 0 and ((d < 0 and 2*floor((1 + d)/2) < w - i0) or (d >= 0 and 2*floor((d)/2) < w - i0)); B[i0] : i0 >= 0 and d < 0 and 2*floor((d)/2) > 2d - w + i0 }"
child:
  schedule: "[d, w] -> [{ A[i0] -> [(i0)]; B[i0] -> [(i0)] }]"
  child:
    sequence:
    - filter: "{ A[x] }"
    - filter: "{ B[x] }"
