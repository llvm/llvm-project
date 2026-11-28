# RUN: llvm-mc --verify -triple=riscv32 --mattr=+xcvmem %s

cv.lb t0, (0), 0
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:11 {{invalid operand for instruction}}
# expected-note@-3:11 {{expected '('}}

cv.lb 0, (t1), 0 # expected-error:7 {{register must be a GPR}}

cv.lb 0, (0), t2 # expected-error:7 {{register must be a GPR}}

cv.lb t0, (t1), -2049
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:17 {{register must be a GPR}}
# expected-note@-3:17 {{immediate must be an integer in the range [-2048, 2047]}}

cv.lb t0, (t1), 2048
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:17 {{register must be a GPR}}
# expected-note@-3:17 {{immediate must be an integer in the range [-2048, 2047]}}

cv.lb t0, (0), t1
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:11 {{invalid operand for instruction}}
# expected-note@-3:11 {{expected '('}}

cv.lb 0, (t1), t1 # expected-error:7 {{register must be a GPR}}

cv.lb t0
# expected-error@-1:9 {{too few operands for instruction}}

cv.lb t0, (t2)
# expected-error@-1:15 {{too few operands for instruction}}

cv.lb t0, (t1), t2, t3 # expected-error:21 {{unexpected extra operand for instruction}}

cv.lbu t0, (0), 0
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:12 {{invalid operand for instruction}}
# expected-note@-3:12 {{expected '('}}

cv.lbu 0, (t1), 0 # expected-error:8 {{register must be a GPR}}

cv.lbu 0, (0), t0 # expected-error:8 {{register must be a GPR}}

cv.lbu t0, (t1), -2049
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:18 {{register must be a GPR}}
# expected-note@-3:18 {{immediate must be an integer in the range [-2048, 2047]}}

cv.lbu t0, (t1), 2048
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:18 {{register must be a GPR}}
# expected-note@-3:18 {{immediate must be an integer in the range [-2048, 2047]}}

cv.lbu t0, (0), t1
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:12 {{invalid operand for instruction}}
# expected-note@-3:12 {{expected '('}}

cv.lbu 0, (t1), t1 # expected-error:8 {{register must be a GPR}}

cv.lbu t0
# expected-error@-1:10 {{too few operands for instruction}}

cv.lbu t0, (t2)
# expected-error@-1:16 {{too few operands for instruction}}

cv.lbu t0, (t1), t2, t3 # expected-error:22 {{unexpected extra operand for instruction}}

cv.lh t0, (0), 0
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:11 {{invalid operand for instruction}}
# expected-note@-3:11 {{expected '('}}

cv.lh 0, (t1), 0 # expected-error:7 {{register must be a GPR}}

cv.lh 0, (0), t2 # expected-error:7 {{register must be a GPR}}

cv.lh t0, (t1), -2049
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:17 {{register must be a GPR}}
# expected-note@-3:17 {{immediate must be an integer in the range [-2048, 2047]}}

cv.lh t0, (t1), 2048
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:17 {{register must be a GPR}}
# expected-note@-3:17 {{immediate must be an integer in the range [-2048, 2047]}}

cv.lh t0, (0), t1
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:11 {{invalid operand for instruction}}
# expected-note@-3:11 {{expected '('}}

cv.lh t0, t1(0) # expected-error:14 {{expected GPR register}}

cv.lh 0, (t1), t1 # expected-error:7 {{register must be a GPR}}

cv.lh t0
# expected-error@-1:9 {{too few operands for instruction}}

cv.lh t0, (t1)
# expected-error@-1:15 {{too few operands for instruction}}

cv.lh t0, (t1), t2, t3 # expected-error:21 {{unexpected extra operand for instruction}}

cv.lhu t0, (0), 0
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:12 {{invalid operand for instruction}}
# expected-note@-3:12 {{expected '('}}

cv.lhu 0, (t1), 0 # expected-error:8 {{register must be a GPR}}

cv.lhu 0, 0(t1) # expected-error:8 {{register must be a GPR}}

cv.lhu t0, (t1), -2049
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:18 {{register must be a GPR}}
# expected-note@-3:18 {{immediate must be an integer in the range [-2048, 2047]}}

cv.lhu t0, (t1), 2048
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:18 {{register must be a GPR}}
# expected-note@-3:18 {{immediate must be an integer in the range [-2048, 2047]}}

cv.lhu t0, (0), t1
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:12 {{invalid operand for instruction}}
# expected-note@-3:12 {{expected '('}}

cv.lhu t0, t1(0) # expected-error:15 {{expected GPR register}}

cv.lhu 0, t0, t1 # expected-error:13 {{expected '(' or invalid operand}}

cv.lhu t0
# expected-error@-1:10 {{too few operands for instruction}}

cv.lhu t0, (t1)
# expected-error@-1:16 {{too few operands for instruction}}

cv.lhu t0, (t1), t2, t3 # expected-error:22 {{unexpected extra operand for instruction}}

cv.lw t0, (0), 0
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:11 {{invalid operand for instruction}}
# expected-note@-3:11 {{expected '('}}

cv.lw 0, (t1), 0 # expected-error:7 {{register must be a GPR}}

cv.lw 0, (0), t2 # expected-error:7 {{register must be a GPR}}

cv.lw t0, (t1), -2049
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:17 {{register must be a GPR}}
# expected-note@-3:17 {{immediate must be an integer in the range [-2048, 2047]}}

cv.lw t0, (t1), 2048
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:17 {{register must be a GPR}}
# expected-note@-3:17 {{immediate must be an integer in the range [-2048, 2047]}}

cv.lw t0, (0), t1
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:11 {{invalid operand for instruction}}
# expected-note@-3:11 {{expected '('}}

cv.lw t0, t1(0) # expected-error:14 {{expected GPR register}}

cv.lw 0, (t0), t1 # expected-error:7 {{register must be a GPR}}

cv.lw t0
# expected-error@-1:9 {{too few operands for instruction}}

cv.lw t0, (t1)
# expected-error@-1:15 {{too few operands for instruction}}

cv.lw t0, (t1), t2, t3 # expected-error:21 {{unexpected extra operand for instruction}}

cv.sb t0, (0), 0
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:11 {{invalid operand for instruction}}
# expected-note@-3:11 {{expected '('}}

cv.sb 0, (t0), 0 # expected-error:7 {{register must be a GPR}}

cv.sb t0, 0(t1)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:11 {{invalid operand for instruction}}
# expected-note@-3:11 {{expected '('}}

cv.sb t0, (t1), 2048
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:17 {{register must be a GPR}}
# expected-note@-3:17 {{immediate must be an integer in the range [-2048, 2047]}}

cv.sb t0, (0), t1
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:11 {{invalid operand for instruction}}
# expected-note@-3:11 {{expected '('}}

cv.sb 0, (t1), t1 # expected-error:7 {{register must be a GPR}}

cv.sb t0
# expected-error@-1:9 {{too few operands for instruction}}

cv.sh t0, (0), 0
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:11 {{invalid operand for instruction}}
# expected-note@-3:11 {{expected '('}}

cv.sh 0, (t1), 0 # expected-error:7 {{register must be a GPR}}

cv.sh t0, 0(t1)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:11 {{invalid operand for instruction}}
# expected-note@-3:11 {{expected '('}}

cv.sh t0, (t1), 2048
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:17 {{register must be a GPR}}
# expected-note@-3:17 {{immediate must be an integer in the range [-2048, 2047]}}

cv.sh t0, (0), t1
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:11 {{invalid operand for instruction}}
# expected-note@-3:11 {{expected '('}}

cv.sh 0, (t1), t1 # expected-error:7 {{register must be a GPR}}

cv.sh t0
# expected-error@-1:9 {{too few operands for instruction}}

cv.sw t0, (0), 0
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:11 {{invalid operand for instruction}}
# expected-note@-3:11 {{expected '('}}

cv.sw 0, (t1), 0 # expected-error:7 {{register must be a GPR}}

cv.sw t0, 0(t1)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:11 {{invalid operand for instruction}}
# expected-note@-3:11 {{expected '('}}

cv.sw t0, (t1), 2048
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:17 {{register must be a GPR}}
# expected-note@-3:17 {{immediate must be an integer in the range [-2048, 2047]}}

cv.sw t0, (0), t1
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:11 {{invalid operand for instruction}}
# expected-note@-3:11 {{expected '('}}

cv.sw 0, (t1), t1 # expected-error:7 {{register must be a GPR}}

cv.sw t0
# expected-error@-1:9 {{too few operands for instruction}}

cv.lb t0, f0(t1) # expected-error:11 {{expected GPR register}}

cv.sb t0, t0(f1) # expected-error:14 {{expected GPR register}}
