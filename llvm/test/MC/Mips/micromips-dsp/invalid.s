# RUN: not llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips32r6 -mattr=micromips -mattr=+dsp 2>%t1
# RUN: FileCheck %s < %t1

  repl.ph $2, -513         # CHECK: :[[@LINE]]:15: error: expected 10-bit signed immediate
  repl.ph $2, 512          # CHECK: :[[@LINE]]:15: error: expected 10-bit signed immediate
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
                           # bposge32 is microMIPS DSP instruction but it is removed in Release 6
  bposge32 342             # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
