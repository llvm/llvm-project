# RUN: not llvm-mc %s -triple=mips-unknown-unknown -show-encoding -mattr=dsp 2>%t1
# RUN: FileCheck %s < %t1

  extp $2, $ac1, -1        # CHECK: :[[@LINE]]:18: error: expected 5-bit unsigned immediate
  extp $2, $ac1, 32        # CHECK: :[[@LINE]]:18: error: expected 5-bit unsigned immediate
  extpdp $2, $ac1, -1      # CHECK: :[[@LINE]]:20: error: expected 5-bit unsigned immediate
  extpdp $2, $ac1, 32      # CHECK: :[[@LINE]]:20: error: expected 5-bit unsigned immediate
  extr.w $2, $ac1, -1      # CHECK: :[[@LINE]]:20: error: expected 5-bit unsigned immediate
  extr.w $2, $ac1, 32      # CHECK: :[[@LINE]]:20: error: expected 5-bit unsigned immediate
  extr_r.w $2, $ac1, -1    # CHECK: :[[@LINE]]:22: error: expected 5-bit unsigned immediate
  extr_r.w $2, $ac1, 32    # CHECK: :[[@LINE]]:22: error: expected 5-bit unsigned immediate
  extr_rs.w $2, $ac1, -1   # CHECK: :[[@LINE]]:23: error: expected 5-bit unsigned immediate
  extr_rs.w $2, $ac1, 32   # CHECK: :[[@LINE]]:23: error: expected 5-bit unsigned immediate
  shll.ph $3, $4, 16       # CHECK: :[[@LINE]]:19: error: expected 4-bit unsigned immediate
  shll.ph $3, $4, -1       # CHECK: :[[@LINE]]:19: error: expected 4-bit unsigned immediate
  shll_s.ph $3, $4, 16     # CHECK: :[[@LINE]]:21: error: expected 4-bit unsigned immediate
  shll_s.ph $3, $4, -1     # CHECK: :[[@LINE]]:21: error: expected 4-bit unsigned immediate
  shll.qb $3, $4, 8        # CHECK: :[[@LINE]]:19: error: expected 3-bit unsigned immediate
  shll.qb $3, $4, -1       # CHECK: :[[@LINE]]:19: error: expected 3-bit unsigned immediate
  shll_s.w $3, $4, 32      # CHECK: :[[@LINE]]:20: error: expected 5-bit unsigned immediate
  shll_s.w $3, $4, -1      # CHECK: :[[@LINE]]:20: error: expected 5-bit unsigned immediate
  shra.ph $3, $4, 16       # CHECK: :[[@LINE]]:19: error: expected 4-bit unsigned immediate
  shra.ph $3, $4, -1       # CHECK: :[[@LINE]]:19: error: expected 4-bit unsigned immediate
  shra_r.ph $3, $4, 16     # CHECK: :[[@LINE]]:21: error: expected 4-bit unsigned immediate
  shra_r.ph $3, $4, -1     # CHECK: :[[@LINE]]:21: error: expected 4-bit unsigned immediate
  shra_r.w $3, $4, 32      # CHECK: :[[@LINE]]:20: error: expected 5-bit unsigned immediate
  shra_r.w $3, $4, -1      # CHECK: :[[@LINE]]:20: error: expected 5-bit unsigned immediate
  shrl.qb $3, $4, 8        # CHECK: :[[@LINE]]:19: error: expected 3-bit unsigned immediate
  shrl.qb $3, $4, -1       # CHECK: :[[@LINE]]:19: error: expected 3-bit unsigned immediate
  shilo $ac1, 64           # CHECK: :[[@LINE]]:15: error: expected 6-bit signed immediate
  shilo $ac1, -64          # CHECK: :[[@LINE]]:15: error: expected 6-bit signed immediate
  repl.qb $2, -1           # CHECK: :[[@LINE]]:15: error: expected 8-bit unsigned immediate
  repl.qb $2, 256          # CHECK: :[[@LINE]]:15: error: expected 8-bit unsigned immediate
  repl.ph $2, -513         # CHECK: :[[@LINE]]:15: error: expected 10-bit signed immediate
  repl.ph $2, 512          # CHECK: :[[@LINE]]:15: error: expected 10-bit signed immediate
  rddsp $2, -1             # CHECK: :[[@LINE]]:13: error: expected 10-bit unsigned immediate
  rddsp $2, 1024           # CHECK: :[[@LINE]]:13: error: expected 10-bit unsigned immediate
  wrdsp $5, -1             # CHECK: :[[@LINE]]:13: error: expected 10-bit unsigned immediate
  wrdsp $5, 1024           # CHECK: :[[@LINE]]:13: error: expected 10-bit unsigned immediate
