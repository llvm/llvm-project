# RUN: not llvm-mc -triple=riscv32 --mattr=+xcvsimd %s 2>&1 \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ERROR

//===----------------------------------------------------------------------===//
// cv.add.h
//===----------------------------------------------------------------------===//

cv.add.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.add.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.add.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.add.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.add.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.add.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.add.b
//===----------------------------------------------------------------------===//

cv.add.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.add.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.add.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.add.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.add.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.add.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.add.sc.h
//===----------------------------------------------------------------------===//

cv.add.sc.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.add.sc.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.add.sc.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.add.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.add.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.add.sc.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.add.sc.b
//===----------------------------------------------------------------------===//

cv.add.sc.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.add.sc.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.add.sc.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.add.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.add.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.add.sc.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.add.sci.h
//===----------------------------------------------------------------------===//

cv.add.sci.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.add.sci.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.add.sci.h t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.add.sci.h t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.add.sci.h t0, t1, 63
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.add.sci.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.add.sci.b
//===----------------------------------------------------------------------===//

cv.add.sci.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.add.sci.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.add.sci.b t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.add.sci.b t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.add.sci.b t0, t1, 63
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.add.sci.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.sub.h
//===----------------------------------------------------------------------===//

cv.sub.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.sub.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.sub.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.sub.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sub.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sub.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.sub.b
//===----------------------------------------------------------------------===//

cv.sub.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.sub.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.sub.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.sub.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sub.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sub.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.sub.sc.h
//===----------------------------------------------------------------------===//

cv.sub.sc.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.sub.sc.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.sub.sc.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.sub.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sub.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sub.sc.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.sub.sc.b
//===----------------------------------------------------------------------===//

cv.sub.sc.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.sub.sc.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.sub.sc.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.sub.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sub.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sub.sc.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.sub.sci.h
//===----------------------------------------------------------------------===//

cv.sub.sci.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.sub.sci.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.sub.sci.h t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.sub.sci.h t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.sub.sci.h t0, t1, 63
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.sub.sci.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.sub.sci.b
//===----------------------------------------------------------------------===//

cv.sub.sci.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.sub.sci.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.sub.sci.b t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.sub.sci.b t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.sub.sci.b t0, t1, 63
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.sub.sci.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.avg.h
//===----------------------------------------------------------------------===//

cv.avg.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.avg.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.avg.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.avg.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.avg.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.avg.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.avg.b
//===----------------------------------------------------------------------===//

cv.avg.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.avg.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.avg.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.avg.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.avg.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.avg.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.avg.sc.h
//===----------------------------------------------------------------------===//

cv.avg.sc.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.avg.sc.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.avg.sc.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.avg.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.avg.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.avg.sc.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.avg.sc.b
//===----------------------------------------------------------------------===//

cv.avg.sc.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.avg.sc.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.avg.sc.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.avg.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.avg.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.avg.sc.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.avg.sci.h
//===----------------------------------------------------------------------===//

cv.avg.sci.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.avg.sci.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.avg.sci.h t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.avg.sci.h t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.avg.sci.h t0, t1, 63
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.avg.sci.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.avg.sci.b
//===----------------------------------------------------------------------===//

cv.avg.sci.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.avg.sci.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.avg.sci.b t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.avg.sci.b t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.avg.sci.b t0, t1, 63
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.avg.sci.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.avgu.h
//===----------------------------------------------------------------------===//

cv.avgu.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.avgu.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.avgu.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.avgu.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.avgu.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.avgu.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.avgu.b
//===----------------------------------------------------------------------===//

cv.avgu.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.avgu.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.avgu.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.avgu.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.avgu.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.avgu.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.avgu.sc.h
//===----------------------------------------------------------------------===//

cv.avgu.sc.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.avgu.sc.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.avgu.sc.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.avgu.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.avgu.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.avgu.sc.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.avgu.sc.b
//===----------------------------------------------------------------------===//

cv.avgu.sc.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.avgu.sc.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.avgu.sc.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.avgu.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.avgu.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.avgu.sc.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.avgu.sci.h
//===----------------------------------------------------------------------===//

cv.avgu.sci.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.avgu.sci.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.avgu.sci.h t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.avgu.sci.h t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.avgu.sci.h t0, t1, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.avgu.sci.h t0, t1, 64
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.avgu.sci.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.avgu.sci.b
//===----------------------------------------------------------------------===//

cv.avgu.sci.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.avgu.sci.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.avgu.sci.b t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.avgu.sci.b t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.avgu.sci.b t0, t1, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.avgu.sci.b t0, t1, 64
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.avgu.sci.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.min.h
//===----------------------------------------------------------------------===//

cv.min.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.min.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.min.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.min.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.min.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.min.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.min.b
//===----------------------------------------------------------------------===//

cv.min.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.min.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.min.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.min.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.min.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.min.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.min.sc.h
//===----------------------------------------------------------------------===//

cv.min.sc.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.min.sc.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.min.sc.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.min.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.min.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.min.sc.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.min.sc.b
//===----------------------------------------------------------------------===//

cv.min.sc.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.min.sc.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.min.sc.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.min.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.min.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.min.sc.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.min.sci.h
//===----------------------------------------------------------------------===//

cv.min.sci.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.min.sci.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.min.sci.h t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.min.sci.h t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.min.sci.h t0, t1, 63
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.min.sci.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.min.sci.b
//===----------------------------------------------------------------------===//

cv.min.sci.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.min.sci.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.min.sci.b t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.min.sci.b t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.min.sci.b t0, t1, 63
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.min.sci.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.minu.h
//===----------------------------------------------------------------------===//

cv.minu.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.minu.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.minu.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.minu.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.minu.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.minu.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.minu.b
//===----------------------------------------------------------------------===//

cv.minu.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.minu.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.minu.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.minu.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.minu.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.minu.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.minu.sc.h
//===----------------------------------------------------------------------===//

cv.minu.sc.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.minu.sc.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.minu.sc.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.minu.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.minu.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.minu.sc.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.minu.sc.b
//===----------------------------------------------------------------------===//

cv.minu.sc.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.minu.sc.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.minu.sc.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.minu.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.minu.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.minu.sc.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.minu.sci.h
//===----------------------------------------------------------------------===//

cv.minu.sci.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.minu.sci.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.minu.sci.h t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.minu.sci.h t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.minu.sci.h t0, t1, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.minu.sci.h t0, t1, 64
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.minu.sci.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.minu.sci.b
//===----------------------------------------------------------------------===//

cv.minu.sci.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.minu.sci.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.minu.sci.b t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.minu.sci.b t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.minu.sci.b t0, t1, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.minu.sci.b t0, t1, 64
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.minu.sci.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.max.h
//===----------------------------------------------------------------------===//

cv.max.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.max.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.max.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.max.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.max.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.max.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.max.b
//===----------------------------------------------------------------------===//

cv.max.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.max.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.max.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.max.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.max.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.max.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.max.sc.h
//===----------------------------------------------------------------------===//

cv.max.sc.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.max.sc.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.max.sc.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.max.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.max.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.max.sc.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.max.sc.b
//===----------------------------------------------------------------------===//

cv.max.sc.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.max.sc.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.max.sc.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.max.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.max.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.max.sc.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.max.sci.h
//===----------------------------------------------------------------------===//

cv.max.sci.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.max.sci.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.max.sci.h t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.max.sci.h t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.max.sci.h t0, t1, 63
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.max.sci.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.max.sci.b
//===----------------------------------------------------------------------===//

cv.max.sci.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.max.sci.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.max.sci.b t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.max.sci.b t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.max.sci.b t0, t1, 63
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.max.sci.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.maxu.h
//===----------------------------------------------------------------------===//

cv.maxu.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.maxu.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.maxu.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.maxu.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.maxu.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.maxu.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.maxu.b
//===----------------------------------------------------------------------===//

cv.maxu.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.maxu.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.maxu.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.maxu.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.maxu.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.maxu.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.maxu.sc.h
//===----------------------------------------------------------------------===//

cv.maxu.sc.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.maxu.sc.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.maxu.sc.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.maxu.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.maxu.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.maxu.sc.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.maxu.sc.b
//===----------------------------------------------------------------------===//

cv.maxu.sc.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.maxu.sc.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.maxu.sc.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.maxu.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.maxu.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.maxu.sc.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.maxu.sci.h
//===----------------------------------------------------------------------===//

cv.maxu.sci.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.maxu.sci.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.maxu.sci.h t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.maxu.sci.h t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.maxu.sci.h t0, t1, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.maxu.sci.h t0, t1, 64
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.maxu.sci.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.maxu.sci.b
//===----------------------------------------------------------------------===//

cv.maxu.sci.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.maxu.sci.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.maxu.sci.b t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.maxu.sci.b t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.maxu.sci.b t0, t1, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.maxu.sci.b t0, t1, 64
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.maxu.sci.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.srl.h
//===----------------------------------------------------------------------===//

cv.srl.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.srl.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.srl.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.srl.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.srl.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.srl.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.srl.b
//===----------------------------------------------------------------------===//

cv.srl.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.srl.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.srl.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.srl.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.srl.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.srl.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.srl.sc.h
//===----------------------------------------------------------------------===//

cv.srl.sc.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.srl.sc.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.srl.sc.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.srl.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.srl.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.srl.sc.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.srl.sc.b
//===----------------------------------------------------------------------===//

cv.srl.sc.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.srl.sc.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.srl.sc.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.srl.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.srl.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.srl.sc.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.srl.sci.h
//===----------------------------------------------------------------------===//

cv.srl.sci.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.srl.sci.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.srl.sci.h t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [0, 15]

cv.srl.sci.h t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [0, 15]

cv.srl.sci.h t0, t1, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 15]

cv.srl.sci.h t0, t1, 16
# CHECK-ERROR: immediate must be an integer in the range [0, 15]

cv.srl.sci.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.srl.sci.b
//===----------------------------------------------------------------------===//

cv.srl.sci.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.srl.sci.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.srl.sci.b t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [0, 7]

cv.srl.sci.b t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [0, 7]

cv.srl.sci.b t0, t1, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 7]

cv.srl.sci.b t0, t1, 8
# CHECK-ERROR: immediate must be an integer in the range [0, 7]

cv.srl.sci.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.sra.h
//===----------------------------------------------------------------------===//

cv.sra.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.sra.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.sra.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.sra.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sra.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sra.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.sra.b
//===----------------------------------------------------------------------===//

cv.sra.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.sra.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.sra.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.sra.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sra.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sra.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.sra.sc.h
//===----------------------------------------------------------------------===//

cv.sra.sc.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.sra.sc.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.sra.sc.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.sra.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sra.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sra.sc.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.sra.sc.b
//===----------------------------------------------------------------------===//

cv.sra.sc.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.sra.sc.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.sra.sc.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.sra.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sra.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sra.sc.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.sra.sci.h
//===----------------------------------------------------------------------===//

cv.sra.sci.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.sra.sci.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.sra.sci.h t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [0, 15]

cv.sra.sci.h t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [0, 15]

cv.sra.sci.h t0, t1, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 15]

cv.sra.sci.h t0, t1, 16
# CHECK-ERROR: immediate must be an integer in the range [0, 15]

cv.sra.sci.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.sra.sci.b
//===----------------------------------------------------------------------===//

cv.sra.sci.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.sra.sci.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.sra.sci.b t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [0, 7]

cv.sra.sci.b t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [0, 7]

cv.sra.sci.b t0, t1, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 7]

cv.sra.sci.b t0, t1, 8
# CHECK-ERROR: immediate must be an integer in the range [0, 7]

cv.sra.sci.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.sll.h
//===----------------------------------------------------------------------===//

cv.sll.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.sll.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.sll.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.sll.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sll.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sll.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.sll.b
//===----------------------------------------------------------------------===//

cv.sll.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.sll.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.sll.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.sll.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sll.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sll.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.sll.sc.h
//===----------------------------------------------------------------------===//

cv.sll.sc.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.sll.sc.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.sll.sc.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.sll.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sll.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sll.sc.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.sll.sc.b
//===----------------------------------------------------------------------===//

cv.sll.sc.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.sll.sc.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.sll.sc.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.sll.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sll.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sll.sc.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.sll.sci.h
//===----------------------------------------------------------------------===//

cv.sll.sci.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.sll.sci.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.sll.sci.h t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [0, 15]

cv.sll.sci.h t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [0, 15]

cv.sll.sci.h t0, t1, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 15]

cv.sll.sci.h t0, t1, 16
# CHECK-ERROR: immediate must be an integer in the range [0, 15]

cv.sll.sci.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.sll.sci.b
//===----------------------------------------------------------------------===//

cv.sll.sci.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.sll.sci.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.sll.sci.b t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [0, 7]

cv.sll.sci.b t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [0, 7]

cv.sll.sci.b t0, t1, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 7]

cv.sll.sci.b t0, t1, 8
# CHECK-ERROR: immediate must be an integer in the range [0, 7]

cv.sll.sci.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.or.h
//===----------------------------------------------------------------------===//

cv.or.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.or.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.or.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.or.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.or.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.or.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.or.b
//===----------------------------------------------------------------------===//

cv.or.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.or.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.or.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.or.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.or.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.or.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.or.sc.h
//===----------------------------------------------------------------------===//

cv.or.sc.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.or.sc.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.or.sc.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.or.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.or.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.or.sc.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.or.sc.b
//===----------------------------------------------------------------------===//

cv.or.sc.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.or.sc.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.or.sc.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.or.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.or.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.or.sc.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.or.sci.h
//===----------------------------------------------------------------------===//

cv.or.sci.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.or.sci.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.or.sci.h t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.or.sci.h t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.or.sci.h t0, t1, 63
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.or.sci.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.or.sci.b
//===----------------------------------------------------------------------===//

cv.or.sci.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.or.sci.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.or.sci.b t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.or.sci.b t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.or.sci.b t0, t1, 63
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.or.sci.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.xor.h
//===----------------------------------------------------------------------===//

cv.xor.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.xor.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.xor.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.xor.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.xor.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.xor.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.xor.b
//===----------------------------------------------------------------------===//

cv.xor.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.xor.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.xor.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.xor.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.xor.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.xor.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.xor.sc.h
//===----------------------------------------------------------------------===//

cv.xor.sc.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.xor.sc.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.xor.sc.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.xor.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.xor.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.xor.sc.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.xor.sc.b
//===----------------------------------------------------------------------===//

cv.xor.sc.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.xor.sc.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.xor.sc.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.xor.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.xor.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.xor.sc.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.xor.sci.h
//===----------------------------------------------------------------------===//

cv.xor.sci.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.xor.sci.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.xor.sci.h t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.xor.sci.h t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.xor.sci.h t0, t1, 63
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.xor.sci.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.xor.sci.b
//===----------------------------------------------------------------------===//

cv.xor.sci.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.xor.sci.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.xor.sci.b t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.xor.sci.b t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.xor.sci.b t0, t1, 63
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.xor.sci.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.and.h
//===----------------------------------------------------------------------===//

cv.and.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.and.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.and.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.and.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.and.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.and.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.and.b
//===----------------------------------------------------------------------===//

cv.and.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.and.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.and.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.and.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.and.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.and.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.and.sc.h
//===----------------------------------------------------------------------===//

cv.and.sc.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.and.sc.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.and.sc.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.and.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.and.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.and.sc.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.and.sc.b
//===----------------------------------------------------------------------===//

cv.and.sc.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.and.sc.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.and.sc.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.and.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.and.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.and.sc.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.and.sci.h
//===----------------------------------------------------------------------===//

cv.and.sci.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.and.sci.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.and.sci.h t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.and.sci.h t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.and.sci.h t0, t1, 63
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.and.sci.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.and.sci.b
//===----------------------------------------------------------------------===//

cv.and.sci.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.and.sci.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.and.sci.b t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.and.sci.b t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.and.sci.b t0, t1, 63
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.and.sci.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.abs.h
//===----------------------------------------------------------------------===//

cv.abs.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.abs.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.abs.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.abs.h t0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.abs.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.abs.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

//===----------------------------------------------------------------------===//
// cv.abs.b
//===----------------------------------------------------------------------===//

cv.abs.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.abs.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.abs.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.abs.b t0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.abs.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.abs.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

//===----------------------------------------------------------------------===//
// cv.dotup.h
//===----------------------------------------------------------------------===//

cv.dotup.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.dotup.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.dotup.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.dotup.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.dotup.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.dotup.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.dotup.b
//===----------------------------------------------------------------------===//

cv.dotup.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.dotup.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.dotup.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.dotup.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.dotup.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.dotup.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.dotup.sc.h
//===----------------------------------------------------------------------===//

cv.dotup.sc.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.dotup.sc.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.dotup.sc.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.dotup.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.dotup.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.dotup.sc.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.dotup.sc.b
//===----------------------------------------------------------------------===//

cv.dotup.sc.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.dotup.sc.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.dotup.sc.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.dotup.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.dotup.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.dotup.sc.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.dotup.sci.h
//===----------------------------------------------------------------------===//

cv.dotup.sci.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.dotup.sci.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.dotup.sci.h t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.dotup.sci.h t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.dotup.sci.h t0, t1, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.dotup.sci.h t0, t1, 64
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.dotup.sci.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.dotup.sci.b
//===----------------------------------------------------------------------===//

cv.dotup.sci.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.dotup.sci.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.dotup.sci.b t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.dotup.sci.b t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.dotup.sci.b t0, t1, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.dotup.sci.b t0, t1, 64
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.dotup.sci.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.dotusp.h
//===----------------------------------------------------------------------===//

cv.dotusp.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.dotusp.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.dotusp.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.dotusp.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.dotusp.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.dotusp.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.dotusp.b
//===----------------------------------------------------------------------===//

cv.dotusp.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.dotusp.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.dotusp.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.dotusp.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.dotusp.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.dotusp.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.dotusp.sc.h
//===----------------------------------------------------------------------===//

cv.dotusp.sc.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.dotusp.sc.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.dotusp.sc.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.dotusp.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.dotusp.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.dotusp.sc.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.dotusp.sc.b
//===----------------------------------------------------------------------===//

cv.dotusp.sc.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.dotusp.sc.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.dotusp.sc.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.dotusp.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.dotusp.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.dotusp.sc.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.dotusp.sci.h
//===----------------------------------------------------------------------===//

cv.dotusp.sci.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.dotusp.sci.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.dotusp.sci.h t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.dotusp.sci.h t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.dotusp.sci.h t0, t1, 63
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.dotusp.sci.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.dotusp.sci.b
//===----------------------------------------------------------------------===//

cv.dotusp.sci.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.dotusp.sci.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.dotusp.sci.b t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.dotusp.sci.b t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.dotusp.sci.b t0, t1, 63
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.dotusp.sci.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.dotsp.h
//===----------------------------------------------------------------------===//

cv.dotsp.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.dotsp.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.dotsp.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.dotsp.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.dotsp.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.dotsp.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.dotsp.b
//===----------------------------------------------------------------------===//

cv.dotsp.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.dotsp.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.dotsp.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.dotsp.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.dotsp.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.dotsp.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.dotsp.sc.h
//===----------------------------------------------------------------------===//

cv.dotsp.sc.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.dotsp.sc.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.dotsp.sc.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.dotsp.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.dotsp.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.dotsp.sc.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.dotsp.sc.b
//===----------------------------------------------------------------------===//

cv.dotsp.sc.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.dotsp.sc.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.dotsp.sc.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.dotsp.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.dotsp.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.dotsp.sc.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.dotsp.sci.h
//===----------------------------------------------------------------------===//

cv.dotsp.sci.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.dotsp.sci.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.dotsp.sci.h t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.dotsp.sci.h t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.dotsp.sci.h t0, t1, 63
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.dotsp.sci.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.dotsp.sci.b
//===----------------------------------------------------------------------===//

cv.dotsp.sci.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.dotsp.sci.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.dotsp.sci.b t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.dotsp.sci.b t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.dotsp.sci.b t0, t1, 63
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.dotsp.sci.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.sdotup.h
//===----------------------------------------------------------------------===//

cv.sdotup.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.sdotup.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.sdotup.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.sdotup.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sdotup.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sdotup.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.sdotup.b
//===----------------------------------------------------------------------===//

cv.sdotup.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.sdotup.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.sdotup.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.sdotup.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sdotup.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sdotup.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.sdotup.sc.h
//===----------------------------------------------------------------------===//

cv.sdotup.sc.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.sdotup.sc.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.sdotup.sc.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.sdotup.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sdotup.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sdotup.sc.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.sdotup.sc.b
//===----------------------------------------------------------------------===//

cv.sdotup.sc.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.sdotup.sc.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.sdotup.sc.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.sdotup.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sdotup.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sdotup.sc.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.sdotup.sci.h
//===----------------------------------------------------------------------===//

cv.sdotup.sci.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.sdotup.sci.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.sdotup.sci.h t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.sdotup.sci.h t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.sdotup.sci.h t0, t1, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.sdotup.sci.h t0, t1, 64
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.sdotup.sci.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.sdotup.sci.b
//===----------------------------------------------------------------------===//

cv.sdotup.sci.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.sdotup.sci.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.sdotup.sci.b t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.sdotup.sci.b t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.sdotup.sci.b t0, t1, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.sdotup.sci.b t0, t1, 64
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.sdotup.sci.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.sdotusp.h
//===----------------------------------------------------------------------===//

cv.sdotusp.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.sdotusp.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.sdotusp.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.sdotusp.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sdotusp.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sdotusp.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.sdotusp.b
//===----------------------------------------------------------------------===//

cv.sdotusp.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.sdotusp.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.sdotusp.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.sdotusp.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sdotusp.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sdotusp.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.sdotusp.sc.h
//===----------------------------------------------------------------------===//

cv.sdotusp.sc.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.sdotusp.sc.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.sdotusp.sc.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.sdotusp.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sdotusp.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sdotusp.sc.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.sdotusp.sc.b
//===----------------------------------------------------------------------===//

cv.sdotusp.sc.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.sdotusp.sc.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.sdotusp.sc.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.sdotusp.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sdotusp.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sdotusp.sc.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.sdotusp.sci.h
//===----------------------------------------------------------------------===//

cv.sdotusp.sci.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.sdotusp.sci.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.sdotusp.sci.h t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.sdotusp.sci.h t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.sdotusp.sci.h t0, t1, 63
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.sdotusp.sci.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.sdotusp.sci.b
//===----------------------------------------------------------------------===//

cv.sdotusp.sci.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.sdotusp.sci.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.sdotusp.sci.b t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.sdotusp.sci.b t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.sdotusp.sci.b t0, t1, 63
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.sdotusp.sci.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.sdotsp.h
//===----------------------------------------------------------------------===//

cv.sdotsp.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.sdotsp.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.sdotsp.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.sdotsp.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sdotsp.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sdotsp.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.sdotsp.b
//===----------------------------------------------------------------------===//

cv.sdotsp.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.sdotsp.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.sdotsp.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.sdotsp.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sdotsp.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sdotsp.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.sdotsp.sc.h
//===----------------------------------------------------------------------===//

cv.sdotsp.sc.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.sdotsp.sc.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.sdotsp.sc.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.sdotsp.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sdotsp.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sdotsp.sc.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.sdotsp.sc.b
//===----------------------------------------------------------------------===//

cv.sdotsp.sc.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.sdotsp.sc.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.sdotsp.sc.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.sdotsp.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sdotsp.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sdotsp.sc.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.sdotsp.sci.h
//===----------------------------------------------------------------------===//

cv.sdotsp.sci.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.sdotsp.sci.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.sdotsp.sci.h t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.sdotsp.sci.h t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.sdotsp.sci.h t0, t1, 63
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.sdotsp.sci.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.sdotsp.sci.b
//===----------------------------------------------------------------------===//

cv.sdotsp.sci.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.sdotsp.sci.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.sdotsp.sci.b t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.sdotsp.sci.b t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.sdotsp.sci.b t0, t1, 63
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.sdotsp.sci.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.extract.h
//===----------------------------------------------------------------------===//

cv.extract.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.extract.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.extract.h t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.extract.h t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.extract.h t0, t1, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.extract.h t0, t1, 64
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.extract.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.extract.b
//===----------------------------------------------------------------------===//

cv.extract.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.extract.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.extract.b t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.extract.b t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.extract.b t0, t1, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.extract.b t0, t1, 64
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.extract.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.extractu.h
//===----------------------------------------------------------------------===//

cv.extractu.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.extractu.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.extractu.h t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.extractu.h t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.extractu.h t0, t1, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.extractu.h t0, t1, 64
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.extractu.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.extractu.b
//===----------------------------------------------------------------------===//

cv.extractu.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.extractu.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.extractu.b t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.extractu.b t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.extractu.b t0, t1, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.extractu.b t0, t1, 64
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.extractu.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.insert.h
//===----------------------------------------------------------------------===//

cv.insert.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.insert.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.insert.h t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.insert.h t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.insert.h t0, t1, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.insert.h t0, t1, 64
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.insert.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.insert.b
//===----------------------------------------------------------------------===//

cv.insert.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.insert.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.insert.b t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.insert.b t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.insert.b t0, t1, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.insert.b t0, t1, 64
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.insert.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.shuffle.h
//===----------------------------------------------------------------------===//

cv.shuffle.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.shuffle.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.shuffle.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.shuffle.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.shuffle.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.shuffle.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.shuffle.b
//===----------------------------------------------------------------------===//

cv.shuffle.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.shuffle.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.shuffle.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.shuffle.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.shuffle.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.shuffle.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.shuffle.sci.h
//===----------------------------------------------------------------------===//

cv.shuffle.sci.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.shuffle.sci.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.shuffle.sci.h t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.shuffle.sci.h t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.shuffle.sci.h t0, t1, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.shuffle.sci.h t0, t1, 64
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.shuffle.sci.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.shuffleI0.sci.b
//===----------------------------------------------------------------------===//

cv.shuffleI0.sci.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.shuffleI0.sci.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.shuffleI0.sci.b t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.shuffleI0.sci.b t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.shuffleI0.sci.b t0, t1, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.shuffleI0.sci.b t0, t1, 64
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.shuffleI0.sci.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.shuffleI1.sci.b
//===----------------------------------------------------------------------===//

cv.shuffleI1.sci.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.shuffleI1.sci.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.shuffleI1.sci.b t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.shuffleI1.sci.b t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.shuffleI1.sci.b t0, t1, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.shuffleI1.sci.b t0, t1, 64
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.shuffleI1.sci.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.shuffleI2.sci.b
//===----------------------------------------------------------------------===//

cv.shuffleI2.sci.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.shuffleI2.sci.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.shuffleI2.sci.b t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.shuffleI2.sci.b t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.shuffleI2.sci.b t0, t1, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.shuffleI2.sci.b t0, t1, 64
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.shuffleI2.sci.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.shuffleI3.sci.b
//===----------------------------------------------------------------------===//

cv.shuffleI3.sci.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.shuffleI3.sci.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.shuffleI3.sci.b t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.shuffleI3.sci.b t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.shuffleI3.sci.b t0, t1, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.shuffleI3.sci.b t0, t1, 64
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.shuffleI3.sci.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.shuffle2.h
//===----------------------------------------------------------------------===//

cv.shuffle2.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.shuffle2.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.shuffle2.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.shuffle2.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.shuffle2.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.shuffle2.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.shuffle2.b
//===----------------------------------------------------------------------===//

cv.shuffle2.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.shuffle2.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.shuffle2.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.shuffle2.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.shuffle2.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.shuffle2.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.pack
//===----------------------------------------------------------------------===//

cv.pack 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.pack t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.pack t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.pack t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.pack t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.pack t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.pack.h
//===----------------------------------------------------------------------===//

cv.pack.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.pack.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.pack.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.pack.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.pack.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.pack.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.packhi.b
//===----------------------------------------------------------------------===//

cv.packhi.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.packhi.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.packhi.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.packhi.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.packhi.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.packhi.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.packlo.b
//===----------------------------------------------------------------------===//

cv.packlo.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.packlo.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.packlo.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.packlo.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.packlo.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.packlo.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmpeq.h
//===----------------------------------------------------------------------===//

cv.cmpeq.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpeq.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpeq.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.cmpeq.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpeq.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpeq.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmpeq.b
//===----------------------------------------------------------------------===//

cv.cmpeq.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpeq.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpeq.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.cmpeq.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpeq.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpeq.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmpeq.sc.h
//===----------------------------------------------------------------------===//

cv.cmpeq.sc.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpeq.sc.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpeq.sc.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.cmpeq.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpeq.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpeq.sc.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmpeq.sc.b
//===----------------------------------------------------------------------===//

cv.cmpeq.sc.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpeq.sc.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpeq.sc.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.cmpeq.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpeq.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpeq.sc.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmpeq.sci.h
//===----------------------------------------------------------------------===//

cv.cmpeq.sci.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpeq.sci.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpeq.sci.h t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.cmpeq.sci.h t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.cmpeq.sci.h t0, t1, 63
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.cmpeq.sci.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmpeq.sci.b
//===----------------------------------------------------------------------===//

cv.cmpeq.sci.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpeq.sci.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpeq.sci.b t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.cmpeq.sci.b t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.cmpeq.sci.b t0, t1, 63
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.cmpeq.sci.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmpne.h
//===----------------------------------------------------------------------===//

cv.cmpne.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpne.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpne.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.cmpne.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpne.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpne.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmpne.b
//===----------------------------------------------------------------------===//

cv.cmpne.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpne.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpne.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.cmpne.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpne.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpne.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmpne.sc.h
//===----------------------------------------------------------------------===//

cv.cmpne.sc.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpne.sc.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpne.sc.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.cmpne.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpne.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpne.sc.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmpne.sc.b
//===----------------------------------------------------------------------===//

cv.cmpne.sc.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpne.sc.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpne.sc.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.cmpne.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpne.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpne.sc.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmpne.sci.h
//===----------------------------------------------------------------------===//

cv.cmpne.sci.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpne.sci.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpne.sci.h t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.cmpne.sci.h t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.cmpne.sci.h t0, t1, 63
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.cmpne.sci.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmpne.sci.b
//===----------------------------------------------------------------------===//

cv.cmpne.sci.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpne.sci.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpne.sci.b t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.cmpne.sci.b t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.cmpne.sci.b t0, t1, 63
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.cmpne.sci.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmpgt.h
//===----------------------------------------------------------------------===//

cv.cmpgt.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpgt.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpgt.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.cmpgt.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpgt.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpgt.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmpgt.b
//===----------------------------------------------------------------------===//

cv.cmpgt.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpgt.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpgt.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.cmpgt.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpgt.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpgt.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmpgt.sc.h
//===----------------------------------------------------------------------===//

cv.cmpgt.sc.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpgt.sc.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpgt.sc.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.cmpgt.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpgt.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpgt.sc.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmpgt.sc.b
//===----------------------------------------------------------------------===//

cv.cmpgt.sc.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpgt.sc.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpgt.sc.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.cmpgt.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpgt.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpgt.sc.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmpgt.sci.h
//===----------------------------------------------------------------------===//

cv.cmpgt.sci.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpgt.sci.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpgt.sci.h t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.cmpgt.sci.h t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.cmpgt.sci.h t0, t1, 63
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.cmpgt.sci.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmpgt.sci.b
//===----------------------------------------------------------------------===//

cv.cmpgt.sci.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpgt.sci.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpgt.sci.b t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.cmpgt.sci.b t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.cmpgt.sci.b t0, t1, 63
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.cmpgt.sci.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmpge.h
//===----------------------------------------------------------------------===//

cv.cmpge.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpge.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpge.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.cmpge.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpge.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpge.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmpge.b
//===----------------------------------------------------------------------===//

cv.cmpge.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpge.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpge.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.cmpge.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpge.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpge.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmpge.sc.h
//===----------------------------------------------------------------------===//

cv.cmpge.sc.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpge.sc.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpge.sc.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.cmpge.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpge.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpge.sc.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmpge.sc.b
//===----------------------------------------------------------------------===//

cv.cmpge.sc.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpge.sc.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpge.sc.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.cmpge.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpge.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpge.sc.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmpge.sci.h
//===----------------------------------------------------------------------===//

cv.cmpge.sci.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpge.sci.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpge.sci.h t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.cmpge.sci.h t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.cmpge.sci.h t0, t1, 63
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.cmpge.sci.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmpge.sci.b
//===----------------------------------------------------------------------===//

cv.cmpge.sci.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpge.sci.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpge.sci.b t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.cmpge.sci.b t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.cmpge.sci.b t0, t1, 63
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.cmpge.sci.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmplt.h
//===----------------------------------------------------------------------===//

cv.cmplt.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmplt.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmplt.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.cmplt.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmplt.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmplt.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmplt.b
//===----------------------------------------------------------------------===//

cv.cmplt.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmplt.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmplt.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.cmplt.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmplt.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmplt.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmplt.sc.h
//===----------------------------------------------------------------------===//

cv.cmplt.sc.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmplt.sc.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmplt.sc.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.cmplt.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmplt.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmplt.sc.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmplt.sc.b
//===----------------------------------------------------------------------===//

cv.cmplt.sc.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmplt.sc.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmplt.sc.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.cmplt.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmplt.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmplt.sc.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmplt.sci.h
//===----------------------------------------------------------------------===//

cv.cmplt.sci.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmplt.sci.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmplt.sci.h t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.cmplt.sci.h t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.cmplt.sci.h t0, t1, 63
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.cmplt.sci.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmplt.sci.b
//===----------------------------------------------------------------------===//

cv.cmplt.sci.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmplt.sci.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmplt.sci.b t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.cmplt.sci.b t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.cmplt.sci.b t0, t1, 63
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.cmplt.sci.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmple.h
//===----------------------------------------------------------------------===//

cv.cmple.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmple.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmple.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.cmple.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmple.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmple.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmple.b
//===----------------------------------------------------------------------===//

cv.cmple.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmple.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmple.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.cmple.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmple.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmple.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmple.sc.h
//===----------------------------------------------------------------------===//

cv.cmple.sc.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmple.sc.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmple.sc.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.cmple.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmple.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmple.sc.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmple.sc.b
//===----------------------------------------------------------------------===//

cv.cmple.sc.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmple.sc.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmple.sc.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.cmple.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmple.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmple.sc.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmple.sci.h
//===----------------------------------------------------------------------===//

cv.cmple.sci.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmple.sci.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmple.sci.h t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.cmple.sci.h t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.cmple.sci.h t0, t1, 63
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.cmple.sci.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmple.sci.b
//===----------------------------------------------------------------------===//

cv.cmple.sci.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmple.sci.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmple.sci.b t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.cmple.sci.b t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.cmple.sci.b t0, t1, 63
# CHECK-ERROR: immediate must be an integer in the range [-32, 31]

cv.cmple.sci.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmpgtu.h
//===----------------------------------------------------------------------===//

cv.cmpgtu.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpgtu.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpgtu.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.cmpgtu.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpgtu.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpgtu.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmpgtu.b
//===----------------------------------------------------------------------===//

cv.cmpgtu.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpgtu.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpgtu.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.cmpgtu.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpgtu.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpgtu.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmpgtu.sc.h
//===----------------------------------------------------------------------===//

cv.cmpgtu.sc.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpgtu.sc.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpgtu.sc.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.cmpgtu.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpgtu.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpgtu.sc.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmpgtu.sc.b
//===----------------------------------------------------------------------===//

cv.cmpgtu.sc.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpgtu.sc.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpgtu.sc.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.cmpgtu.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpgtu.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpgtu.sc.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmpgtu.sci.h
//===----------------------------------------------------------------------===//

cv.cmpgtu.sci.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpgtu.sci.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpgtu.sci.h t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.cmpgtu.sci.h t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.cmpgtu.sci.h t0, t1, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.cmpgtu.sci.h t0, t1, 64
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.cmpgtu.sci.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmpgtu.sci.b
//===----------------------------------------------------------------------===//

cv.cmpgtu.sci.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpgtu.sci.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpgtu.sci.b t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.cmpgtu.sci.b t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.cmpgtu.sci.b t0, t1, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.cmpgtu.sci.b t0, t1, 64
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.cmpgtu.sci.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmpgeu.h
//===----------------------------------------------------------------------===//

cv.cmpgeu.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpgeu.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpgeu.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.cmpgeu.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpgeu.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpgeu.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmpgeu.b
//===----------------------------------------------------------------------===//

cv.cmpgeu.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpgeu.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpgeu.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.cmpgeu.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpgeu.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpgeu.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmpgeu.sc.h
//===----------------------------------------------------------------------===//

cv.cmpgeu.sc.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpgeu.sc.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpgeu.sc.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.cmpgeu.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpgeu.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpgeu.sc.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmpgeu.sc.b
//===----------------------------------------------------------------------===//

cv.cmpgeu.sc.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpgeu.sc.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpgeu.sc.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.cmpgeu.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpgeu.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpgeu.sc.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmpgeu.sci.h
//===----------------------------------------------------------------------===//

cv.cmpgeu.sci.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpgeu.sci.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpgeu.sci.h t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.cmpgeu.sci.h t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.cmpgeu.sci.h t0, t1, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.cmpgeu.sci.h t0, t1, 64
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.cmpgeu.sci.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmpgeu.sci.b
//===----------------------------------------------------------------------===//

cv.cmpgeu.sci.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpgeu.sci.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpgeu.sci.b t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.cmpgeu.sci.b t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.cmpgeu.sci.b t0, t1, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.cmpgeu.sci.b t0, t1, 64
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.cmpgeu.sci.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmpltu.h
//===----------------------------------------------------------------------===//

cv.cmpltu.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpltu.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpltu.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.cmpltu.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpltu.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpltu.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmpltu.b
//===----------------------------------------------------------------------===//

cv.cmpltu.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpltu.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpltu.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.cmpltu.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpltu.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpltu.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmpltu.sc.h
//===----------------------------------------------------------------------===//

cv.cmpltu.sc.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpltu.sc.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpltu.sc.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.cmpltu.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpltu.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpltu.sc.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmpltu.sc.b
//===----------------------------------------------------------------------===//

cv.cmpltu.sc.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpltu.sc.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpltu.sc.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.cmpltu.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpltu.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpltu.sc.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmpltu.sci.h
//===----------------------------------------------------------------------===//

cv.cmpltu.sci.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpltu.sci.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpltu.sci.h t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.cmpltu.sci.h t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.cmpltu.sci.h t0, t1, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.cmpltu.sci.h t0, t1, 64
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.cmpltu.sci.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmpltu.sci.b
//===----------------------------------------------------------------------===//

cv.cmpltu.sci.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpltu.sci.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpltu.sci.b t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.cmpltu.sci.b t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.cmpltu.sci.b t0, t1, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.cmpltu.sci.b t0, t1, 64
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.cmpltu.sci.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmpleu.h
//===----------------------------------------------------------------------===//

cv.cmpleu.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpleu.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpleu.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.cmpleu.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpleu.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpleu.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmpleu.b
//===----------------------------------------------------------------------===//

cv.cmpleu.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpleu.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpleu.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.cmpleu.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpleu.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpleu.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmpleu.sc.h
//===----------------------------------------------------------------------===//

cv.cmpleu.sc.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpleu.sc.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpleu.sc.h t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.cmpleu.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpleu.sc.h t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpleu.sc.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmpleu.sc.b
//===----------------------------------------------------------------------===//

cv.cmpleu.sc.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpleu.sc.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpleu.sc.b t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.cmpleu.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpleu.sc.b t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cmpleu.sc.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmpleu.sci.h
//===----------------------------------------------------------------------===//

cv.cmpleu.sci.h 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpleu.sci.h t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpleu.sci.h t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.cmpleu.sci.h t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.cmpleu.sci.h t0, t1, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.cmpleu.sci.h t0, t1, 64
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.cmpleu.sci.h t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cmpleu.sci.b
//===----------------------------------------------------------------------===//

cv.cmpleu.sci.b 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpleu.sci.b t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cmpleu.sci.b t0, t1, t2, t3
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.cmpleu.sci.b t0, t1, t2
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.cmpleu.sci.b t0, t1, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.cmpleu.sci.b t0, t1, 64
# CHECK-ERROR: immediate must be an integer in the range [0, 63]

cv.cmpleu.sci.b t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cplxmul.r
//===----------------------------------------------------------------------===//

cv.cplxmul.r 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cplxmul.r t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cplxmul.r t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.cplxmul.r t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cplxmul.r t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cplxmul.r t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cplxmul.i
//===----------------------------------------------------------------------===//

cv.cplxmul.i 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cplxmul.i t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cplxmul.i t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.cplxmul.i t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cplxmul.i t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cplxmul.i t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cplxmul.r.div2
//===----------------------------------------------------------------------===//

cv.cplxmul.r.div2 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cplxmul.r.div2 t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cplxmul.r.div2 t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.cplxmul.r.div2 t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cplxmul.r.div2 t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cplxmul.r.div2 t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cplxmul.i.div2
//===----------------------------------------------------------------------===//

cv.cplxmul.i.div2 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cplxmul.i.div2 t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cplxmul.i.div2 t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.cplxmul.i.div2 t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cplxmul.i.div2 t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cplxmul.i.div2 t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cplxmul.r.div4
//===----------------------------------------------------------------------===//

cv.cplxmul.r.div4 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cplxmul.r.div4 t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cplxmul.r.div4 t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.cplxmul.r.div4 t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cplxmul.r.div4 t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cplxmul.r.div4 t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cplxmul.i.div4
//===----------------------------------------------------------------------===//

cv.cplxmul.i.div4 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cplxmul.i.div4 t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cplxmul.i.div4 t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.cplxmul.i.div4 t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cplxmul.i.div4 t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cplxmul.i.div4 t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cplxmul.r.div8
//===----------------------------------------------------------------------===//

cv.cplxmul.r.div8 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cplxmul.r.div8 t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cplxmul.r.div8 t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.cplxmul.r.div8 t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cplxmul.r.div8 t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cplxmul.r.div8 t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cplxmul.i.div8
//===----------------------------------------------------------------------===//

cv.cplxmul.i.div8 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cplxmul.i.div8 t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cplxmul.i.div8 t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.cplxmul.i.div8 t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cplxmul.i.div8 t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cplxmul.i.div8 t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.cplxconj
//===----------------------------------------------------------------------===//

cv.cplxconj 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cplxconj t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.cplxconj t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.cplxconj t0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.cplxconj t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.cplxconj t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

//===----------------------------------------------------------------------===//
// cv.subrotmj
//===----------------------------------------------------------------------===//

cv.subrotmj 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.subrotmj t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.subrotmj t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.subrotmj t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.subrotmj t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.subrotmj t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.subrotmj.div2
//===----------------------------------------------------------------------===//

cv.subrotmj.div2 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.subrotmj.div2 t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.subrotmj.div2 t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.subrotmj.div2 t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.subrotmj.div2 t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.subrotmj.div2 t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.subrotmj.div4
//===----------------------------------------------------------------------===//

cv.subrotmj.div4 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.subrotmj.div4 t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.subrotmj.div4 t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.subrotmj.div4 t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.subrotmj.div4 t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.subrotmj.div4 t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.subrotmj.div8
//===----------------------------------------------------------------------===//

cv.subrotmj.div8 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.subrotmj.div8 t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.subrotmj.div8 t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.subrotmj.div8 t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.subrotmj.div8 t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.subrotmj.div8 t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.add.div2
//===----------------------------------------------------------------------===//

cv.add.div2 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.add.div2 t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.add.div2 t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.add.div2 t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.add.div2 t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.add.div2 t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.add.div4
//===----------------------------------------------------------------------===//

cv.add.div4 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.add.div4 t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.add.div4 t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.add.div4 t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.add.div4 t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.add.div4 t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.add.div8
//===----------------------------------------------------------------------===//

cv.add.div8 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.add.div8 t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.add.div8 t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.add.div8 t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.add.div8 t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.add.div8 t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.sub.div2
//===----------------------------------------------------------------------===//

cv.sub.div2 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.sub.div2 t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.sub.div2 t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.sub.div2 t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sub.div2 t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sub.div2 t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.sub.div4
//===----------------------------------------------------------------------===//

cv.sub.div4 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.sub.div4 t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.sub.div4 t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.sub.div4 t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sub.div4 t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sub.div4 t0, t1
# CHECK-ERROR: too few operands for instruction

//===----------------------------------------------------------------------===//
// cv.sub.div8
//===----------------------------------------------------------------------===//

cv.sub.div8 0, t1, t2
# CHECK-ERROR: invalid operand for instruction

cv.sub.div8 t0, 0, t2
# CHECK-ERROR: invalid operand for instruction

cv.sub.div8 t0, t1, t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.sub.div8 t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sub.div8 t0, t1, 0
# CHECK-ERROR: invalid operand for instruction

cv.sub.div8 t0, t1
# CHECK-ERROR: too few operands for instruction

