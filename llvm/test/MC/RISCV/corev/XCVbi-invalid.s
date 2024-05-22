# RUN: not llvm-mc -triple=riscv32 --mattr=+xcvbi %s 2>&1 \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ERROR

//===----------------------------------------------------------------------===//
// cv.beqimm
//===----------------------------------------------------------------------===//

cv.beqimm 0, 0, 0
# CHECK-ERROR: invalid operand for instruction

cv.beqimm t0, t1, 0
# CHECK-ERROR: immediate must be an integer in the range [-16, 15]

cv.beqimm t0, 0, t1
# CHECK-ERROR: immediate must be a multiple of 2 bytes in the range [-4096, 4094]

cv.beqimm t0, 16, 0
# CHECK-ERROR: immediate must be an integer in the range [-16, 15]

cv.beqimm t0, -17, 0
# CHECK-ERROR: immediate must be an integer in the range [-16, 15]

cv.beqimm t0, 0, 1
# CHECK-ERROR: immediate must be a multiple of 2 bytes in the range [-4096, 4094]

cv.beqimm t0, 0, 4096
# CHECK-ERROR: immediate must be a multiple of 2 bytes in the range [-4096, 4094]

cv.beqimm t0, 0, -4098
# CHECK-ERROR: immediate must be a multiple of 2 bytes in the range [-4096, 4094]

//===----------------------------------------------------------------------===//
// cv.bneimm
//===----------------------------------------------------------------------===//

cv.bneimm 0, 0, 0
# CHECK-ERROR: invalid operand for instruction

cv.bneimm t0, t1, 0
# CHECK-ERROR: immediate must be an integer in the range [-16, 15]

cv.bneimm t0, 0, t1
# CHECK-ERROR: immediate must be a multiple of 2 bytes in the range [-4096, 4094]

cv.bneimm t0, 16, 0
# CHECK-ERROR: immediate must be an integer in the range [-16, 15]

cv.bneimm t0, -17, 0
# CHECK-ERROR: immediate must be an integer in the range [-16, 15]

cv.bneimm t0, 0, 1
# CHECK-ERROR: immediate must be a multiple of 2 bytes in the range [-4096, 4094]

cv.bneimm t0, 0, 4096
# CHECK-ERROR: immediate must be a multiple of 2 bytes in the range [-4096, 4094]

cv.bneimm t0, 0, -4098
# CHECK-ERROR: immediate must be a multiple of 2 bytes in the range [-4096, 4094]