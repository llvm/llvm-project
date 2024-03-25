# RUN: not llvm-mc -triple=riscv32 --mattr=+xcvelw %s 2>&1 \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ERROR

cv.elw t0, 0(0)
# CHECK-ERROR: expected register

cv.elw 0, 0(x6)
# CHECK-ERROR: invalid operand for instruction

cv.elw x12, 2048(x6)
# CHECK-ERROR: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo modifier or an integer in the range [-2048, 2047]

cv.elw x12, x1(2047)
# CHECK-ERROR: unexpected token

cv.elw 0, x12(x6)
# CHECK-ERROR: unexpected token

cv.elw x12, x12(x6)
# CHECK-ERROR: unexpected token

cv.elw 0, 0(x6)
# CHECK-ERROR: invalid operand for instruction

cv.elw x0
# CHECK-ERROR: too few operands for instruction
