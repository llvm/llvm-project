# RUN: llvm-mc --verify -triple=riscv32 --mattr=+xcvbi %s

//===----------------------------------------------------------------------===//
// cv.beqimm
//===----------------------------------------------------------------------===//

cv.beqimm 0, 0, 0 # expected-error {{register must be a GPR}}

cv.beqimm t0, t1, 0 # expected-error {{immediate must be an integer in the range [-16, 15]}}

cv.beqimm t0, 0, t1 # expected-error {{immediate must be a multiple of 2 bytes in the range [-4096, 4094]}}

cv.beqimm t0, 16, 0 # expected-error {{immediate must be an integer in the range [-16, 15]}}

cv.beqimm t0, -17, 0 # expected-error {{immediate must be an integer in the range [-16, 15]}}

cv.beqimm t0, 0, 1 # expected-error {{immediate must be a multiple of 2 bytes in the range [-4096, 4094]}}

cv.beqimm t0, 0, 4096 # expected-error {{immediate must be a multiple of 2 bytes in the range [-4096, 4094]}}

cv.beqimm t0, 0, -4098 # expected-error {{immediate must be a multiple of 2 bytes in the range [-4096, 4094]}}

//===----------------------------------------------------------------------===//
// cv.bneimm
//===----------------------------------------------------------------------===//

cv.bneimm 0, 0, 0 # expected-error {{register must be a GPR}}

cv.bneimm t0, t1, 0 # expected-error {{immediate must be an integer in the range [-16, 15]}}

cv.bneimm t0, 0, t1 # expected-error {{immediate must be a multiple of 2 bytes in the range [-4096, 4094]}}

cv.bneimm t0, 16, 0 # expected-error {{immediate must be an integer in the range [-16, 15]}}

cv.bneimm t0, -17, 0 # expected-error {{immediate must be an integer in the range [-16, 15]}}

cv.bneimm t0, 0, 1 # expected-error {{immediate must be a multiple of 2 bytes in the range [-4096, 4094]}}

cv.bneimm t0, 0, 4096 # expected-error {{immediate must be a multiple of 2 bytes in the range [-4096, 4094]}}

cv.bneimm t0, 0, -4098 # expected-error {{immediate must be a multiple of 2 bytes in the range [-4096, 4094]}}