# RUN: not llvm-mc -triple=riscv32 --mattr=+xcvbitmanip %s 2>&1 \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ERROR

cv.extract t0, t1
# CHECK-ERROR: too few operands for instruction

cv.extract t0, t1, 0
# CHECK-ERROR: too few operands for instruction

cv.extract t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.extract t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.extract t0, t1, 0, 32
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.extract t0, t1, 0, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.extract t0, t1, 32, 0
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.extract t0, t1, -1, 0
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.extractu t0, t1
# CHECK-ERROR: too few operands for instruction

cv.extractu t0, t1, 0
# CHECK-ERROR: too few operands for instruction

cv.extractu t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.extractu t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.extractu t0, t1, 0, 32
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.extractu t0, t1, 0, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.extractu t0, t1, 32, 0
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.extractu t0, t1, -1, 0
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.insert t0, t1
# CHECK-ERROR: too few operands for instruction

cv.insert t0, t1, 0
# CHECK-ERROR: too few operands for instruction

cv.insert t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.insert t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.insert t0, t1, 0, 32
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.insert t0, t1, 0, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.insert t0, t1, 32, 0
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.insert t0, t1, -1, 0
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.bclr t0, t1
# CHECK-ERROR: too few operands for instruction

cv.bclr t0, t1, 0
# CHECK-ERROR: too few operands for instruction

cv.bclr t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.bclr t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.bclr t0, t1, 0, 32
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.bclr t0, t1, 0, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.bclr t0, t1, 32, 0
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.bclr t0, t1, -1, 0
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.bset t0, t1
# CHECK-ERROR: too few operands for instruction

cv.bset t0, t1, 0
# CHECK-ERROR: too few operands for instruction

cv.bset t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.bset t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.bset t0, t1, 0, 32
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.bset t0, t1, 0, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.bset t0, t1, 32, 0
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.bset t0, t1, -1, 0
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.bitrev t0, t1
# CHECK-ERROR: too few operands for instruction

cv.bitrev t0, t1, 0
# CHECK-ERROR: too few operands for instruction

cv.bitrev t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [0, 3]

cv.bitrev t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [0, 3]

cv.bitrev t0, t1, 0, 32
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.bitrev t0, t1, 0, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 31]

cv.bitrev t0, t1, 32, 0
# CHECK-ERROR: immediate must be an integer in the range [0, 3]

cv.bitrev t0, t1, -1, 0
# CHECK-ERROR: immediate must be an integer in the range [0, 3]

cv.extractr t0
# CHECK-ERROR: too few operands for instruction

cv.extractr t0, 0
# CHECK-ERROR: invalid operand for instruction

cv.extractr t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.extractr t0, t1
# CHECK-ERROR: too few operands for instruction

cv.extractur t0
# CHECK-ERROR: too few operands for instruction

cv.extractur t0, 0
# CHECK-ERROR: invalid operand for instruction

cv.extractur t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.extractur t0, t1
# CHECK-ERROR: too few operands for instruction

cv.insertr t0
# CHECK-ERROR: too few operands for instruction

cv.insertr t0, 0
# CHECK-ERROR: invalid operand for instruction

cv.insertr t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.insertr t0, t1
# CHECK-ERROR: too few operands for instruction

cv.bclrr t0
# CHECK-ERROR: too few operands for instruction

cv.bclrr t0, 0
# CHECK-ERROR: invalid operand for instruction

cv.bclrr t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.bclrr t0, t1
# CHECK-ERROR: too few operands for instruction

cv.bsetr t0
# CHECK-ERROR: too few operands for instruction

cv.bsetr t0, 0
# CHECK-ERROR: invalid operand for instruction

cv.bsetr t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.bsetr t0, t1
# CHECK-ERROR: too few operands for instruction

cv.ror t0
# CHECK-ERROR: too few operands for instruction

cv.ror t0, 0
# CHECK-ERROR: invalid operand for instruction

cv.ror t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.ror t0, t1
# CHECK-ERROR: too few operands for instruction

cv.ff1 t0
# CHECK-ERROR: too few operands for instruction

cv.ff1 t0, 0
# CHECK-ERROR: invalid operand for instruction

cv.ff1 t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.ff1 t0, t1, t2
# CHECK-ERROR: invalid operand for instruction 

cv.fl1 t0
# CHECK-ERROR: too few operands for instruction

cv.fl1 t0, 0
# CHECK-ERROR: invalid operand for instruction

cv.fl1 t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.fl1 t0, t1, t2
# CHECK-ERROR: invalid operand for instruction 

cv.clb t0
# CHECK-ERROR: too few operands for instruction

cv.clb t0, 0
# CHECK-ERROR: invalid operand for instruction

cv.clb t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.clb t0, t1, t2
# CHECK-ERROR: invalid operand for instruction 

cv.cnt t0
# CHECK-ERROR: too few operands for instruction

cv.cnt t0, 0
# CHECK-ERROR: invalid operand for instruction

cv.cnt t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cnt t0, t1, t2
# CHECK-ERROR: invalid operand for instruction 

