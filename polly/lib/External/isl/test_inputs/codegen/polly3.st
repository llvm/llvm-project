# A Polly generated test case that would result in a failure
# to generate an AST in an earlier version of isl.
domain: "[h, d, f] -> { S[] : 3*floor((h)/3) < h and (f = 0 or d = 0) }"
child:
  context: "[h, d, f] -> { [] : d < 0 || d > 0 }"
