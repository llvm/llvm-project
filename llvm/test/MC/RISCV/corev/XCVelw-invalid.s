# RUN: llvm-mc --verify -triple=riscv32 --mattr=+xcvelw %s

cv.elw t0, 0(0)
# expected-error@-1 {{expected register}}

cv.elw 0, 0(x6)
# expected-error@-1 {{register must be a GPR}}

cv.elw x12, 2048(x6)
# expected-error@-1 {{operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]}}

cv.elw x12, x1(2047)
# expected-error@-1 {{unexpected token}}

cv.elw 0, x12(x6)
# expected-error@-1 {{unexpected token}}

cv.elw x12, x12(x6)
# expected-error@-1 {{unexpected token}}

cv.elw 0, 0(x6)
# expected-error@-1 {{register must be a GPR}}

cv.elw x0
# expected-error@-1 {{too few operands for instruction}}

cv.elw x0, a_symbol
# expected-error@-1 {{register must be a GPR excluding zero (x0)}}
