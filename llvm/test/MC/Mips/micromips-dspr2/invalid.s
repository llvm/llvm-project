# RUN: not llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips32r6 -mattr=micromips -mattr=+dspr2 2>%t1
# RUN: FileCheck %s < %t1

  balign $2, $3, -1        # CHECK: :[[@LINE]]:18: error: expected 2-bit unsigned immediate
  balign $2, $3, 4         # CHECK: :[[@LINE]]:18: error: expected 2-bit unsigned immediate
  shra.qb $3, $4, 8        # CHECK: :[[@LINE]]:19: error: expected 3-bit unsigned immediate
  shra.qb $3, $4, -1       # CHECK: :[[@LINE]]:19: error: expected 3-bit unsigned immediate
  shra_r.qb $3, $4, 8      # CHECK: :[[@LINE]]:21: error: expected 3-bit unsigned immediate
  shra_r.qb $3, $4, -1     # CHECK: :[[@LINE]]:21: error: expected 3-bit unsigned immediate
  shrl.ph $3, $4, 16       # CHECK: :[[@LINE]]:19: error: expected 4-bit unsigned immediate
  shrl.ph $3, $4, -1       # CHECK: :[[@LINE]]:19: error: expected 4-bit unsigned immediate
  append $3, $4, 32           # CHECK: :[[@LINE]]:18: error: expected 5-bit unsigned immediate
  append $3, $4, -1           # CHECK: :[[@LINE]]:18: error: expected 5-bit unsigned immediate
  mulsa.w.ph $8, $3, $2       # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
  mulsa.w.ph $31, $3, $2      # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
  mulsaq_s.w.ph $8, $3, $2    # CHECK: :[[@LINE]]:17: error: invalid operand for instruction
  mulsaq_s.w.ph $31, $3, $2   # CHECK: :[[@LINE]]:17: error: invalid operand for instruction
                              # bposge32 is microMIPS DSP instruction but it is removed in Release 6
  bposge32 342                # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
