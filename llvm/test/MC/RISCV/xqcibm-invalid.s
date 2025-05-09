# Xqcibm - Qualcomm uC Bit Manipulation Extension
# RUN: not llvm-mc -triple riscv32 -mattr=+experimental-xqcibm < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-PLUS %s
# RUN: not llvm-mc -triple riscv32 -mattr=-experimental-xqcibm < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-MINUS %s

# CHECK-PLUS: :[[@LINE+2]]:18: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:18: error: invalid operand for instruction
qc.compress2 x7, 5

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.compress2 x7

# CHECK-PLUS: :[[@LINE+2]]:14: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:14: error: invalid operand for instruction
qc.compress2 x0,x5

# CHECK-PLUS: :[[@LINE+2]]:18: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:18: error: invalid operand for instruction
qc.compress2 x7, x0

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)
qc.compress2 x7, x5


# CHECK-PLUS: :[[@LINE+2]]:19: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:19: error: invalid operand for instruction
qc.compress3 x10, 22

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.compress3 x10

# CHECK-PLUS: :[[@LINE+2]]:14: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:14: error: invalid operand for instruction
qc.compress3 x0, x22

# CHECK-PLUS: :[[@LINE+2]]:19: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:19: error: invalid operand for instruction
qc.compress3 x10, x0

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)
qc.compress3 x10, x22


# CHECK-PLUS: :[[@LINE+2]]:17: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:17: error: invalid operand for instruction
qc.expand2 x23, 23

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.expand2 x23

# CHECK-PLUS: :[[@LINE+2]]:12: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:12: error: invalid operand for instruction
qc.expand2 x0, x23

# CHECK-PLUS: :[[@LINE+2]]:17: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:17: error: invalid operand for instruction
qc.expand2 x23, x0

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)
qc.expand2 x23, x23


# CHECK-PLUS: :[[@LINE+2]]:16: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:16: error: invalid operand for instruction
qc.expand3 x2, 6

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.expand3 x2

# CHECK-PLUS: :[[@LINE+2]]:12: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:12: error: invalid operand for instruction
qc.expand3 x0, x6

# CHECK-PLUS: :[[@LINE+2]]:16: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:16: error: invalid operand for instruction
qc.expand3 x2, x0

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)
qc.expand3 x2, x6


# CHECK-PLUS: :[[@LINE+2]]:13: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:13: error: invalid operand for instruction
qc.clo x23, 24

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.clo x23

# CHECK-PLUS: :[[@LINE+2]]:8: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:8: error: invalid operand for instruction
qc.clo x0, x24

# CHECK-PLUS: :[[@LINE+2]]:13: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:13: error: invalid operand for instruction
qc.clo x23, x0

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)
qc.clo x23, x24


# CHECK-PLUS: :[[@LINE+2]]:13: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:13: error: invalid operand for instruction
qc.cto x12, 13

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.cto x12

# CHECK-PLUS: :[[@LINE+2]]:8: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:8: error: invalid operand for instruction
qc.cto x0, x13

# CHECK-PLUS: :[[@LINE+2]]:13: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:13: error: invalid operand for instruction
qc.cto x12, x0

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)
qc.cto x12, x13


# CHECK-PLUS: :[[@LINE+2]]:16: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:16: error: invalid operand for instruction
qc.brev32 x20, 24

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.brev32 x20

# CHECK-PLUS: :[[@LINE+2]]:11: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:11: error: invalid operand for instruction
qc.brev32 x0, x24

# CHECK-PLUS: :[[@LINE+2]]:16: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:16: error: invalid operand for instruction
qc.brev32 x20, x0

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)
qc.brev32 x20, x24


# CHECK-PLUS: :[[@LINE+2]]:16: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:16: error: invalid operand for instruction
qc.insbri x10, 20, -1024

# CHECK-PLUS: :[[@LINE+2]]:11: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:11: error: invalid operand for instruction
qc.insbri x0, x20, -1024

# CHECK-PLUS: :[[@LINE+2]]:16: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:16: error: invalid operand for instruction
qc.insbri x10, x0, -1024

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.insbri x10, x20

# CHECK-PLUS: :[[@LINE+1]]:21: error: immediate must be an integer in the range [-1024, 1023]
qc.insbri x10, x20, -1027

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)
qc.insbri x10, x20, -1024


# CHECK-PLUS: :[[@LINE+2]]:10: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:10: error: invalid operand for instruction
qc.insbi x0, -10, 12, 15

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.insbi x6, -10, 12

# CHECK-PLUS: :[[@LINE+1]]:14: error: immediate must be an integer in the range [-16, 15]
qc.insbi x6, -17, 12, 15

# CHECK-PLUS: :[[@LINE+1]]:19: error: immediate must be an integer in the range [1, 32]
qc.insbi x6, -10, 45, 15

# CHECK-PLUS: :[[@LINE+1]]:23: error: immediate must be an integer in the range [0, 31]
qc.insbi x6, -10, 12, 65

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)
qc.insbi x6, -10, 12, 15


# CHECK: :[[@LINE+1]]:14: error: invalid operand for instruction
qc.insb x10, 7, 6, 31

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.insb x10, x7, 6

# CHECK-PLUS: :[[@LINE+2]]:9: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:9: error: invalid operand for instruction
qc.insb x0, x7, 6, 31

# CHECK-PLUS: :[[@LINE+1]]:18: error: immediate must be an integer in the range [1, 32]
qc.insb x10, x7, 46, 31

# CHECK-PLUS: :[[@LINE+1]]:21: error: immediate must be an integer in the range [0, 31]
qc.insb x10, x7, 6, 61

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)
qc.insb x10, x7, 6, 31


# CHECK: :[[@LINE+1]]:15: error: invalid operand for instruction
qc.insbh x20, 12, 8, 12

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.insbh x20, x12, 8

# CHECK-PLUS: :[[@LINE+2]]:10: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:10: error: invalid operand for instruction
qc.insbh x0, x12, 8, 12

# CHECK-PLUS: :[[@LINE+1]]:20: error: immediate must be an integer in the range [1, 32]
qc.insbh x20, x12, 48, 12

# CHECK-PLUS: :[[@LINE+1]]:23: error: immediate must be an integer in the range [0, 31]
qc.insbh x20, x12, 8, 72

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)
qc.insbh x20, x12, 8, 12


# CHECK-PLUS: :[[@LINE+2]]:14: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:14: error: invalid operand for instruction
qc.extu x15, 12, 20, 20

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.extu x15, x12, 20

# CHECK-PLUS: :[[@LINE+2]]:9: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:9: error: invalid operand for instruction
qc.extu x0, x12, 20, 20

# CHECK-PLUS: :[[@LINE+2]]:14: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:14: error: invalid operand for instruction
qc.extu x15, x0, 20, 20

# CHECK-PLUS: :[[@LINE+1]]:19: error: immediate must be an integer in the range [1, 32]
qc.extu x15, x12, 0, 20

# CHECK-PLUS: :[[@LINE+1]]:23: error: immediate must be an integer in the range [0, 31]
qc.extu x15, x12, 20, 60

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)
qc.extu x15, x12, 20, 20


# CHECK-PLUS: :[[@LINE+2]]:13: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:13: error: invalid operand for instruction
qc.ext x27, 6, 31, 1

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.ext x27, x6, 31

# CHECK-PLUS: :[[@LINE+2]]:8: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:8: error: invalid operand for instruction
qc.ext x0, x6, 31, 1

# CHECK-PLUS: :[[@LINE+2]]:13: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:13: error: invalid operand for instruction
qc.ext x27, x0, 31, 1

# CHECK-PLUS: :[[@LINE+1]]:17: error: immediate must be an integer in the range [1, 32]
qc.ext x27, x6, 0, 1

# CHECK-PLUS: :[[@LINE+1]]:21: error: immediate must be an integer in the range [0, 31]
qc.ext x27, x6, 31, 41

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)
qc.ext x27, x6, 31, 1


# CHECK-PLUS: :[[@LINE+2]]:14: error: register must be a GPR excluding t6 (x31)
# CHECK-MINUS: :[[@LINE+1]]:14: error: invalid operand for instruction
qc.extdu x1, 8, 8, 8

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.extdu x1, x8, 8

# CHECK-PLUS: :[[@LINE+2]]:10: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:10: error: invalid operand for instruction
qc.extdu x0, x8, 8, 8

# CHECK-PLUS: :[[@LINE+1]]:18: error: immediate must be an integer in the range [1, 32]
qc.extdu x1, x8, 48, 8

# CHECK-PLUS: :[[@LINE+1]]:21: error: immediate must be an integer in the range [0, 31]
qc.extdu x1, x8, 8, 78

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)
qc.extdu x1, x8, 8, 8


# CHECK-PLUS: :[[@LINE+2]]:14: error: register must be a GPR excluding t6 (x31)
# CHECK-MINUS: :[[@LINE+1]]:14: error: invalid operand for instruction
qc.extd x13, 21, 10, 15

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.extd x13, x21, 10

# CHECK-PLUS: :[[@LINE+2]]:9: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:9: error: invalid operand for instruction
qc.extd x0, x21, 10, 15

# CHECK-PLUS: :[[@LINE+1]]:19: error: immediate must be an integer in the range [1, 32]
qc.extd x13, x21, 60, 15

# CHECK-PLUS: :[[@LINE+1]]:23: error: immediate must be an integer in the range [0, 31]
qc.extd x13, x21, 10, 85

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)
qc.extd x13, x21, 10, 15


# CHECK-PLUS: :[[@LINE+2]]:20: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:20: error: invalid operand for instruction
qc.insbr x10, x19, 5

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.insbr x10, x20

# CHECK-PLUS: :[[@LINE+2]]:10: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:10: error: invalid operand for instruction
qc.insbr x0, x19, x5

# CHECK-PLUS: :[[@LINE+2]]:20: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:20: error: invalid operand for instruction
qc.insbr x10, x19, x0

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)
qc.insbr x10, x19, x5


# CHECK-PLUS: :[[@LINE+2]]:20: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:20: error: invalid operand for instruction
qc.insbhr x15, x4, 6

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.insbhr x15, x4

# CHECK-PLUS: :[[@LINE+2]]:11: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:11: error: invalid operand for instruction
qc.insbhr x0, x4, x6

# CHECK-PLUS: :[[@LINE+2]]:20: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:20: error: invalid operand for instruction
qc.insbhr x15, x4, x0

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)
qc.insbhr x15, x4, x6


# CHECK-PLUS: :[[@LINE+2]]:20: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:20: error: invalid operand for instruction
qc.insbpr x21, x8, 9

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.insbpr x21, x8

# CHECK-PLUS: :[[@LINE+2]]:11: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:11: error: invalid operand for instruction
qc.insbpr x0, x8, x9

# CHECK-PLUS: :[[@LINE+2]]:20: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:20: error: invalid operand for instruction
qc.insbpr x21, x8, x0

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)
qc.insbpr x21, x8, x9


# CHECK-PLUS: :[[@LINE+2]]:20: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:20: error: invalid operand for instruction
qc.insbprh x2, x3, 11

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.insbprh x2, x3

# CHECK-PLUS: :[[@LINE+2]]:12: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:12: error: invalid operand for instruction
qc.insbprh x0, x3, x11

# CHECK-PLUS: :[[@LINE+2]]:20: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:20: error: invalid operand for instruction
qc.insbprh x2, x3, x0

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)
qc.insbprh x2, x3, x11


# CHECK-PLUS: :[[@LINE+2]]:20: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:20: error: invalid operand for instruction
qc.extdur x9, x19, 29

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.extdur x9, x19

# CHECK-PLUS: :[[@LINE+2]]:11: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:11: error: invalid operand for instruction
qc.extdur x0, x19, x29

# CHECK-PLUS: :[[@LINE+2]]:15: error: register must be a GPR excluding t6 (x31)
# CHECK-MINUS: :[[@LINE+1]]:15: error: invalid operand for instruction
qc.extdur x9, x31, x29

# CHECK-PLUS: :[[@LINE+2]]:20: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:20: error: invalid operand for instruction
qc.extdur x9, x19, x0

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)
qc.extdur x9, x19, x29


# CHECK-PLUS: :[[@LINE+2]]:20: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:20: error: invalid operand for instruction
qc.extdr x12, x29, 30

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.extdr x12, x29

# CHECK-PLUS: :[[@LINE+2]]:10: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:10: error: invalid operand for instruction
qc.extdr x0, x29, x30

# CHECK-PLUS: :[[@LINE+2]]:15: error: register must be a GPR excluding t6 (x31)
# CHECK-MINUS: :[[@LINE+1]]:15: error: invalid operand for instruction
qc.extdr x12, x31, x30

# CHECK-PLUS: :[[@LINE+2]]:20: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:20: error: invalid operand for instruction
qc.extdr x12, x29, x0

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)
qc.extdr x12, x29, x30


# CHECK-PLUS: :[[@LINE+2]]:22: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:22: error: invalid operand for instruction
qc.extdupr x13, x23, 3

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.extdupr x13, x23

# CHECK-PLUS: :[[@LINE+2]]:12: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:12: error: invalid operand for instruction
qc.extdupr x0, x23, x3

# CHECK-PLUS: :[[@LINE+2]]:17: error: register must be a GPR excluding t6 (x31)
# CHECK-MINUS: :[[@LINE+1]]:17: error: invalid operand for instruction
qc.extdupr x13, x31, x3

# CHECK-PLUS: :[[@LINE+2]]:22: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:22: error: invalid operand for instruction
qc.extdupr x13, x23, x0

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)
qc.extdupr x13, x23, x3


# CHECK-PLUS: :[[@LINE+2]]:22: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:22: error: invalid operand for instruction
qc.extduprh x18, x8, 9

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.extduprh x18, x8

# CHECK-PLUS: :[[@LINE+2]]:13: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:13: error: invalid operand for instruction
qc.extduprh x0, x8, x9

# CHECK-PLUS: :[[@LINE+2]]:18: error: register must be a GPR excluding t6 (x31)
# CHECK-MINUS: :[[@LINE+1]]:18: error: invalid operand for instruction
qc.extduprh x18, x31, x9

# CHECK-PLUS: :[[@LINE+2]]:22: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:22: error: invalid operand for instruction
qc.extduprh x18, x8, x0

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)
qc.extduprh x18, x8, x9


# CHECK-PLUS: :[[@LINE+2]]:19: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:19: error: invalid operand for instruction
qc.extdpr x1, x4, 15

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.extdpr x1, x4

# CHECK-PLUS: :[[@LINE+2]]:11: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:11: error: invalid operand for instruction
qc.extdpr x0, x4, x15

# CHECK-PLUS: :[[@LINE+2]]:15: error: register must be a GPR excluding t6 (x31)
# CHECK-MINUS: :[[@LINE+1]]:15: error: invalid operand for instruction
qc.extdpr x1, x31, x15

# CHECK-PLUS: :[[@LINE+2]]:19: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:19: error: invalid operand for instruction
qc.extdpr x1, x4, x0

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)
qc.extdpr x1, x4, x15


# CHECK-PLUS: :[[@LINE+2]]:21: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:21: error: invalid operand for instruction
qc.extdprh x6, x24, 25

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.extdprh x6, x24

# CHECK-PLUS: :[[@LINE+2]]:12: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:12: error: invalid operand for instruction
qc.extdprh x0, x24, x25

# CHECK-PLUS: :[[@LINE+2]]:16: error: register must be a GPR excluding t6 (x31)
# CHECK-MINUS: :[[@LINE+1]]:16: error: invalid operand for instruction
qc.extdprh x6, x31, x25

# CHECK-PLUS: :[[@LINE+2]]:21: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:21: error: invalid operand for instruction
qc.extdprh x6, x24, x0

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)
qc.extdprh x6, x24, x25


# CHECK: :[[@LINE+1]]:12: error: invalid operand for instruction
qc.c.bexti x1, 8

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.c.bexti x15

# CHECK-PLUS: :[[@LINE+1]]:17: error: immediate must be an integer in the range [1, 31]
qc.c.bexti x15, 43

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)
qc.c.bexti x9, 8


# CHECK: :[[@LINE+1]]:12: error: invalid operand for instruction
qc.c.bseti x2, 10

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.c.bseti x12

# CHECK-PLUS: :[[@LINE+1]]:17: error: immediate must be an integer in the range [1, 31]
qc.c.bseti x12, -10

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)
qc.c.bseti x12, 30


# CHECK-PLUS: :[[@LINE+2]]:11: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:11: error: invalid operand for instruction
qc.c.extu x0, 10

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.c.extu x5

# CHECK-PLUS: :[[@LINE+1]]:16: error: immediate must be an integer in the range [6, 32]
qc.c.extu x17, 3

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)
qc.c.extu x17, 32
