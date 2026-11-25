# RUN: llvm-mc --verify -triple=riscv32 --mattr=+xcvbitmanip %s

cv.extract t0, t1 # expected-error {{too few operands for instruction}}

cv.extract t0, t1, 0 # expected-error {{too few operands for instruction}}

cv.extract t0, t1, t2 # expected-error {{immediate must be an integer in the range [0, 31]}}

cv.extract t0, t1, t2, t3 # expected-error {{immediate must be an integer in the range [0, 31]}}

cv.extract t0, t1, 0, 32 # expected-error {{immediate must be an integer in the range [0, 31]}}

cv.extract t0, t1, 0, -1 # expected-error {{immediate must be an integer in the range [0, 31]}}

cv.extract t0, t1, 32, 0 # expected-error {{immediate must be an integer in the range [0, 31]}}

cv.extract t0, t1, -1, 0 # expected-error {{immediate must be an integer in the range [0, 31]}}

cv.extractu t0, t1 # expected-error {{too few operands for instruction}}

cv.extractu t0, t1, 0 # expected-error {{too few operands for instruction}}

cv.extractu t0, t1, t2 # expected-error {{immediate must be an integer in the range [0, 31]}}

cv.extractu t0, t1, t2, t3 # expected-error {{immediate must be an integer in the range [0, 31]}}

cv.extractu t0, t1, 0, 32 # expected-error {{immediate must be an integer in the range [0, 31]}}

cv.extractu t0, t1, 0, -1 # expected-error {{immediate must be an integer in the range [0, 31]}}

cv.extractu t0, t1, 32, 0 # expected-error {{immediate must be an integer in the range [0, 31]}}

cv.extractu t0, t1, -1, 0 # expected-error {{immediate must be an integer in the range [0, 31]}}

cv.insert t0, t1 # expected-error {{too few operands for instruction}}

cv.insert t0, t1, 0 # expected-error {{too few operands for instruction}}

cv.insert t0, t1, t2 # expected-error {{immediate must be an integer in the range [0, 31]}}

cv.insert t0, t1, t2, t3 # expected-error {{immediate must be an integer in the range [0, 31]}}

cv.insert t0, t1, 0, 32 # expected-error {{immediate must be an integer in the range [0, 31]}}

cv.insert t0, t1, 0, -1 # expected-error {{immediate must be an integer in the range [0, 31]}}

cv.insert t0, t1, 32, 0 # expected-error {{immediate must be an integer in the range [0, 31]}}

cv.insert t0, t1, -1, 0 # expected-error {{immediate must be an integer in the range [0, 31]}}

cv.insert a0, a1, 16, 16 # expected-error:19 {{the sum of the immediate operands must be less than 32}}

cv.insert a0, a1, 31, 31 # expected-error:19 {{the sum of the immediate operands must be less than 32}}

cv.insert a0, a1, 1, 31 # expected-error:19 {{the sum of the immediate operands must be less than 32}}

cv.insert a0, a1, 31, 1 # expected-error:19 {{the sum of the immediate operands must be less than 32}}

cv.bclr t0, t1 # expected-error {{too few operands for instruction}}

cv.bclr t0, t1, 0 # expected-error {{too few operands for instruction}}

cv.bclr t0, t1, t2 # expected-error {{immediate must be an integer in the range [0, 31]}}

cv.bclr t0, t1, t2, t3 # expected-error {{immediate must be an integer in the range [0, 31]}}

cv.bclr t0, t1, 0, 32 # expected-error {{immediate must be an integer in the range [0, 31]}}

cv.bclr t0, t1, 0, -1 # expected-error {{immediate must be an integer in the range [0, 31]}}

cv.bclr t0, t1, 32, 0 # expected-error {{immediate must be an integer in the range [0, 31]}}

cv.bclr t0, t1, -1, 0 # expected-error {{immediate must be an integer in the range [0, 31]}}

cv.bset t0, t1 # expected-error {{too few operands for instruction}}

cv.bset t0, t1, 0 # expected-error {{too few operands for instruction}}

cv.bset t0, t1, t2 # expected-error {{immediate must be an integer in the range [0, 31]}}

cv.bset t0, t1, t2, t3 # expected-error {{immediate must be an integer in the range [0, 31]}}

cv.bset t0, t1, 0, 32 # expected-error {{immediate must be an integer in the range [0, 31]}}

cv.bset t0, t1, 0, -1 # expected-error {{immediate must be an integer in the range [0, 31]}}

cv.bset t0, t1, 32, 0 # expected-error {{immediate must be an integer in the range [0, 31]}}

cv.bset t0, t1, -1, 0 # expected-error {{immediate must be an integer in the range [0, 31]}}

cv.bitrev t0, t1 # expected-error {{too few operands for instruction}}

cv.bitrev t0, t1, 0 # expected-error {{too few operands for instruction}}

cv.bitrev t0, t1, t2 # expected-error {{immediate must be an integer in the range [0, 3]}}

cv.bitrev t0, t1, t2, t3 # expected-error {{immediate must be an integer in the range [0, 3]}}

cv.bitrev t0, t1, 0, 32 # expected-error {{immediate must be an integer in the range [0, 31]}}

cv.bitrev t0, t1, 0, -1 # expected-error {{immediate must be an integer in the range [0, 31]}}

cv.bitrev t0, t1, 32, 0 # expected-error {{immediate must be an integer in the range [0, 3]}}

cv.bitrev t0, t1, -1, 0 # expected-error {{immediate must be an integer in the range [0, 3]}}

cv.extractr t0 # expected-error {{too few operands for instruction}}

cv.extractr t0, 0 # expected-error {{register must be a GPR}}

cv.extractr t0, t1, 0 # expected-error {{register must be a GPR}}

cv.extractr t0, t1 # expected-error {{too few operands for instruction}}

cv.extractur t0 # expected-error {{too few operands for instruction}}

cv.extractur t0, 0 # expected-error {{register must be a GPR}}

cv.extractur t0, t1, 0 # expected-error {{register must be a GPR}}

cv.extractur t0, t1 # expected-error {{too few operands for instruction}}

cv.insertr t0 # expected-error {{too few operands for instruction}}

cv.insertr t0, 0 # expected-error {{register must be a GPR}}

cv.insertr t0, t1, 0 # expected-error {{register must be a GPR}}

cv.insertr t0, t1 # expected-error {{too few operands for instruction}}

cv.bclrr t0 # expected-error {{too few operands for instruction}}

cv.bclrr t0, 0 # expected-error {{register must be a GPR}}

cv.bclrr t0, t1, 0 # expected-error {{register must be a GPR}}

cv.bclrr t0, t1 # expected-error {{too few operands for instruction}}

cv.bsetr t0 # expected-error {{too few operands for instruction}}

cv.bsetr t0, 0 # expected-error {{register must be a GPR}}

cv.bsetr t0, t1, 0 # expected-error {{register must be a GPR}}

cv.bsetr t0, t1 # expected-error {{too few operands for instruction}}

cv.ror t0 # expected-error {{too few operands for instruction}}

cv.ror t0, 0 # expected-error {{register must be a GPR}}

cv.ror t0, t1, 0 # expected-error {{register must be a GPR}}

cv.ror t0, t1 # expected-error {{too few operands for instruction}}

cv.ff1 t0 # expected-error {{too few operands for instruction}}

cv.ff1 t0, 0 # expected-error {{register must be a GPR}}

cv.ff1 t0, t1, 0 # expected-error {{unexpected extra operand for instruction}}

cv.ff1 t0, t1, t2 # expected-error {{unexpected extra operand for instruction}}

cv.fl1 t0 # expected-error {{too few operands for instruction}}

cv.fl1 t0, 0 # expected-error {{register must be a GPR}}

cv.fl1 t0, t1, 0 # expected-error {{unexpected extra operand for instruction}}

cv.fl1 t0, t1, t2 # expected-error {{unexpected extra operand for instruction}}

cv.clb t0 # expected-error {{too few operands for instruction}}

cv.clb t0, 0 # expected-error {{register must be a GPR}}

cv.clb t0, t1, 0 # expected-error {{unexpected extra operand for instruction}}

cv.clb t0, t1, t2 # expected-error {{unexpected extra operand for instruction}}

cv.cnt t0 # expected-error {{too few operands for instruction}}

cv.cnt t0, 0 # expected-error {{register must be a GPR}}

cv.cnt t0, t1, 0 # expected-error {{unexpected extra operand for instruction}}

cv.cnt t0, t1, t2 # expected-error {{unexpected extra operand for instruction}}

