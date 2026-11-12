# A Polly generated test case that would cause a failure
# to handle void expression in an earlier version of isl.
domain: "[e] -> { Stmt_for_body9_us94_last[i0] : 0 <= i0 <= 1; Stmt_cond_false30_us97[i0] : e = 0 and 0 <= i0 <= 1 }"
child:
  context: "[e] -> { [] : (e > 0 and e <= 127) or (e >= -128 and e < 0) }"
  child:
    schedule: "[e] -> [{ Stmt_for_body9_us94_last[i0] -> [(i0)]; Stmt_cond_false30_us97[i0] -> [(i0)] }]"
    child:
      sequence:
      - filter: "[e] -> { Stmt_for_body9_us94_last[i0] }"
      - filter: "[e] -> { Stmt_cond_false30_us97[i0] }"

