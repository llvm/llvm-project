# RUN: not llvm-mc -triple=riscv32 --mattr=+xcvmem %s 2>&1 \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ERROR

cv.lb t0, (0), 0
# CHECK-ERROR: operands must be register and register

cv.lb 0, (t1), 0
# CHECK-ERROR: invalid operand for instruction

cv.lb 0, (0), t2
# CHECK-ERROR: invalid operand for instruction

cv.lb t0, (t1), -2049
# CHECK-ERROR: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo modifier or an integer in the range [-2048, 2047]

cv.lb t0, (t1), 2048
# CHECK-ERROR: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo modifier or an integer in the range [-2048, 2047]

cv.lb t0, (0), t1
# CHECK-ERROR: operands must be register and register

cv.lb 0, (t1), t1
# CHECK-ERROR: invalid operand for instruction

cv.lb t0
# CHECK-ERROR: too few operands for instruction

cv.lb t0, (t2)
# CHECK-ERROR: too few operands for instruction

cv.lb t0, (t1), t2, t3
# CHECK-ERROR: invalid operand for instruction 

cv.lbu t0, (0), 0
# CHECK-ERROR: operands must be register and register

cv.lbu 0, (t1), 0
# CHECK-ERROR: invalid operand for instruction

cv.lbu 0, (0), t0 
# CHECK-ERROR: invalid operand for instruction

cv.lbu t0, (t1), -2049
# CHECK-ERROR: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo modifier or an integer in the range [-2048, 2047]

cv.lbu t0, (t1), 2048
# CHECK-ERROR: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo modifier or an integer in the range [-2048, 2047]

cv.lbu t0, (0), t1
# CHECK-ERROR: operands must be register and register

cv.lbu 0, (t1), t1
# CHECK-ERROR: invalid operand for instruction

cv.lbu t0
# CHECK-ERROR: too few operands for instruction

cv.lbu t0, (t2)
# CHECK-ERROR: too few operands for instruction

cv.lbu t0, (t1), t2, t3 
# CHECK-ERROR: invalid operand for instruction

cv.lh t0, (0), 0
# CHECK-ERROR: operands must be register and register

cv.lh 0, (t1), 0
# CHECK-ERROR: invalid operand for instruction

cv.lh 0, (0), t2
# CHECK-ERROR: invalid operand for instruction

cv.lh t0, (t1), -2049
# CHECK-ERROR: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo modifier or an integer in the range [-2048, 2047]

cv.lh t0, (t1), 2048
# CHECK-ERROR: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo modifier or an integer in the range [-2048, 2047]

cv.lh t0, (0), t1
# CHECK-ERROR: operands must be register and register

cv.lh t0, t1(0)
# CHECK-ERROR: expected register

cv.lh 0, (t1), t1
# CHECK-ERROR: invalid operand for instruction

cv.lh t0
# CHECK-ERROR: too few operands for instruction

cv.lh t0, (t1)
# CHECK-ERROR: too few operands for instruction

cv.lh t0, (t1), t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.lhu t0, (0), 0
# CHECK-ERROR: operands must be register and register

cv.lhu 0, (t1), 0
# CHECK-ERROR: invalid operand for instruction

cv.lhu 0, 0(t1)
# CHECK-ERROR: invalid operand for instruction

cv.lhu t0, (t1), -2049
# CHECK-ERROR: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo modifier or an integer in the range [-2048, 2047]

cv.lhu t0, (t1), 2048
# CHECK-ERROR: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo modifier or an integer in the range [-2048, 2047]

cv.lhu t0, (0), t1
# CHECK-ERROR: operands must be register and register 

cv.lhu t0, t1(0)
# CHECK-ERROR: expected register

cv.lhu 0, t0, t1
# CHECK-ERROR: expected '(' or invalid operand

cv.lhu t0
# CHECK-ERROR: too few operands for instruction

cv.lhu t0, (t1)
# CHECK-ERROR: too few operands for instruction

cv.lhu t0, (t1), t2, t3
# CHECK-ERROR: invalid operand for instruction

cv.lw t0, (0), 0
# CHECK-ERROR: operands must be register and register

cv.lw 0, (t1), 0
# CHECK-ERROR: invalid operand for instruction

cv.lw 0, (0), t2
# CHECK-ERROR: invalid operand for instruction

cv.lw t0, (t1), -2049
# CHECK-ERROR: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo modifier or an integer in the range [-2048, 2047]

cv.lw t0, (t1), 2048
# CHECK-ERROR: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo modifier or an integer in the range [-2048, 2047]

cv.lw t0, (0), t1
# CHECK-ERROR: operands must be register and register

cv.lw t0, t1(0)
# CHECK-ERROR: expected register

cv.lw 0, (t0), t1
# CHECK-ERROR: invalid operand for instruction

cv.lw t0
# CHECK-ERROR: too few operands for instruction

cv.lw t0, (t1)
# CHECK-ERROR: too few operands for instruction

cv.lw t0, (t1), t2, t3
# CHECK-ERROR: invalid operand for instruction 

cv.sb t0, (0), 0
# CHECK-ERROR: operands must be register and register

cv.sb 0, (t0), 0
# CHECK-ERROR: invalid operand for instruction

cv.sb t0, 0(t1)
# CHECK-ERROR: operands must be register and register

cv.sb t0, (t1), 2048
# CHECK-ERROR: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo modifier or an integer in the range [-2048, 2047]

cv.sb t0, (0), t1
# CHECK-ERROR: operands must be register and register

cv.sb 0, (t1), t1
# CHECK-ERROR: invalid operand for instruction

cv.sb t0
# CHECK-ERROR: too few operands for instruction

cv.sh t0, (0), 0
# CHECK-ERROR: operands must be register and register

cv.sh 0, (t1), 0
# CHECK-ERROR: invalid operand for instruction

cv.sh t0, 0(t1)
# CHECK-ERROR: operands must be register and register

cv.sh t0, (t1), 2048
# CHECK-ERROR: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo modifier or an integer in the range [-2048, 2047]

cv.sh t0, (0), t1
# CHECK-ERROR: operands must be register and register

cv.sh 0, (t1), t1
# CHECK-ERROR: invalid operand for instruction

cv.sh t0
# CHECK-ERROR: too few operands for instruction

cv.sw t0, (0), 0
# CHECK-ERROR: operands must be register and register

cv.sw 0, (t1), 0
# CHECK-ERROR: invalid operand for instruction

cv.sw t0, 0(t1)
# CHECK-ERROR: operands must be register and register

cv.sw t0, (t1), 2048
# CHECK-ERROR: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo modifier or an integer in the range [-2048, 2047]

cv.sw t0, (0), t1
# CHECK-ERROR: operands must be register and register

cv.sw 0, (t1), t1
# CHECK-ERROR: invalid operand for instruction

cv.sw t0
# CHECK-ERROR: too few operands for instruction
