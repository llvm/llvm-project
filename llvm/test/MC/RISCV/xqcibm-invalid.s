# Xqcibm - Qualcomm uC Bit Manipulation Extension
# RUN: llvm-mc --verify=plus -triple riscv32 -mattr=+xqcibm < %s
# RUN: llvm-mc --verify=minus -triple riscv32 -mattr=-xqcibm < %s

qc.compress2 x7, 5
# plus-error@-1:18 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.compress2 x7
# plus-error@-1:16 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.compress2 x0,x5
# plus-error@-1:14 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.compress2 x7, x0
# plus-error@-1:18 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.compress2 x7, x5 # minus-error:1 {{instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)}}


qc.compress3 x10, 22
# plus-error@-1:19 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.compress3 x10
# plus-error@-1:17 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.compress3 x0, x22
# plus-error@-1:14 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.compress3 x10, x0
# plus-error@-1:19 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.compress3 x10, x22 # minus-error:1 {{instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)}}


qc.expand2 x23, 23
# plus-error@-1:17 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.expand2 x23
# plus-error@-1:15 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.expand2 x0, x23
# plus-error@-1:12 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.expand2 x23, x0
# plus-error@-1:17 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.expand2 x23, x23 # minus-error:1 {{instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)}}


qc.expand3 x2, 6
# plus-error@-1:16 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.expand3 x2
# plus-error@-1:14 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.expand3 x0, x6
# plus-error@-1:12 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.expand3 x2, x0
# plus-error@-1:16 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.expand3 x2, x6 # minus-error:1 {{instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)}}


qc.clo x23, 24
# plus-error@-1:13 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.clo x23
# plus-error@-1:11 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.clo x0, x24
# plus-error@-1:8 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.clo x23, x0
# plus-error@-1:13 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.clo x23, x24 # minus-error:1 {{instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)}}


qc.cto x12, 13
# plus-error@-1:13 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.cto x12
# plus-error@-1:11 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.cto x0, x13
# plus-error@-1:8 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.cto x12, x0
# plus-error@-1:13 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.cto x12, x13 # minus-error:1 {{instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)}}


qc.brev32 x20, 24
# plus-error@-1:16 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.brev32 x20
# plus-error@-1:14 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.brev32 x0, x24
# plus-error@-1:11 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.brev32 x20, x0
# plus-error@-1:16 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.brev32 x20, x24 # minus-error:1 {{instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)}}


qc.insbri x10, 20, -1024
# plus-error@-1:16 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.insbri x0, x20, -1024
# plus-error@-1:11 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.insbri x10, x0, -1024
# plus-error@-1:16 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.insbri x10, x20
# plus-error@-1:19 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.insbri x10, x20, -1027
# plus-error@-1:21 {{immediate must be an integer in the range [-1024, 1023]}}
# minus-error@-2:1 {{invalid instruction}}

qc.insbri x10, x20, -1024 # minus-error:1 {{instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)}}


qc.insbi x0, -10, 12, 15
# plus-error@-1:10 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.insbi x6, -10, 12
# plus-error@-1:21 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.insbi x6, -17, 12, 15
# plus-error@-1:14 {{immediate must be an integer in the range [-16, 15]}}
# minus-error@-2:1 {{invalid instruction}}

qc.insbi x6, -10, 45, 15
# plus-error@-1:19 {{immediate must be an integer in the range [1, 32]}}
# minus-error@-2:1 {{invalid instruction}}

qc.insbi x6, -10, 12, 65
# plus-error@-1:23 {{immediate must be an integer in the range [0, 31]}}
# minus-error@-2:1 {{invalid instruction}}

qc.insbi x6, -10, 12, 15 # minus-error:1 {{instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)}}


qc.insb x10, 7, 6, 31
# plus-error@-1:14 {{register must be a GPR}}
# minus-error@-2:1 {{invalid instruction}}

qc.insb x10, x7, 6
# plus-error@-1:19 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.insb x0, x7, 6, 31
# plus-error@-1:9 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.insb x10, x7, 46, 31
# plus-error@-1:18 {{immediate must be an integer in the range [1, 32]}}
# minus-error@-2:1 {{invalid instruction}}

qc.insb x10, x7, 6, 61
# plus-error@-1:21 {{immediate must be an integer in the range [0, 31]}}
# minus-error@-2:1 {{invalid instruction}}

qc.insb x10, x7, 6, 31 # minus-error:1 {{instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)}}


qc.insbh x20, 12, 8, 12
# plus-error@-1:15 {{register must be a GPR}}
# minus-error@-2:1 {{invalid instruction}}

qc.insbh x20, x12, 8
# plus-error@-1:21 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.insbh x0, x12, 8, 12
# plus-error@-1:10 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.insbh x20, x12, 48, 12
# plus-error@-1:20 {{immediate must be an integer in the range [1, 32]}}
# minus-error@-2:1 {{invalid instruction}}

qc.insbh x20, x12, 8, 72
# plus-error@-1:23 {{immediate must be an integer in the range [0, 31]}}
# minus-error@-2:1 {{invalid instruction}}

qc.insbh x20, x12, 8, 12 # minus-error:1 {{instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)}}


qc.extu x15, 12, 20, 20
# plus-error@-1:14 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.extu x15, x12, 20
# plus-error@-1:21 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.extu x0, x12, 20, 20
# plus-error@-1:9 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.extu x15, x0, 20, 20
# plus-error@-1:14 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.extu x15, x12, 0, 20
# plus-error@-1:19 {{immediate must be an integer in the range [1, 32]}}
# minus-error@-2:1 {{invalid instruction}}

qc.extu x15, x12, 20, 60
# plus-error@-1:23 {{immediate must be an integer in the range [0, 31]}}
# minus-error@-2:1 {{invalid instruction}}

qc.extu x15, x12, 20, 20 # minus-error:1 {{instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)}}


qc.ext x27, 6, 31, 1
# plus-error@-1:13 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.ext x27, x6, 31
# plus-error@-1:19 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.ext x0, x6, 31, 1
# plus-error@-1:8 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.ext x27, x0, 31, 1
# plus-error@-1:13 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.ext x27, x6, 0, 1
# plus-error@-1:17 {{immediate must be an integer in the range [1, 32]}}
# minus-error@-2:1 {{invalid instruction}}

qc.ext x27, x6, 31, 41
# plus-error@-1:21 {{immediate must be an integer in the range [0, 31]}}
# minus-error@-2:1 {{invalid instruction}}

qc.ext x27, x6, 31, 1 # minus-error:1 {{instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)}}


qc.extdu x1, 8, 8, 8
# plus-error@-1:14 {{register must be a GPR excluding t6 (x31)}}
# minus-error@-2:1 {{invalid instruction}}

qc.extdu x1, x8, 8
# plus-error@-1:19 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.extdu x0, x8, 8, 8
# plus-error@-1:10 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.extdu x1, x8, 48, 8
# plus-error@-1:18 {{immediate must be an integer in the range [1, 32]}}
# minus-error@-2:1 {{invalid instruction}}

qc.extdu x1, x8, 8, 78
# plus-error@-1:21 {{immediate must be an integer in the range [0, 31]}}
# minus-error@-2:1 {{invalid instruction}}

qc.extdu x1, x8, 8, 8 # minus-error:1 {{instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)}}


qc.extd x13, 21, 10, 15
# plus-error@-1:14 {{register must be a GPR excluding t6 (x31)}}
# minus-error@-2:1 {{invalid instruction}}

qc.extd x13, x21, 10
# plus-error@-1:21 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.extd x0, x21, 10, 15
# plus-error@-1:9 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.extd x13, x21, 60, 15
# plus-error@-1:19 {{immediate must be an integer in the range [1, 32]}}
# minus-error@-2:1 {{invalid instruction}}

qc.extd x13, x21, 10, 85
# plus-error@-1:23 {{immediate must be an integer in the range [0, 31]}}
# minus-error@-2:1 {{invalid instruction}}

qc.extd x13, x21, 10, 15 # minus-error:1 {{instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)}}


qc.insbr x10, x19, 5
# plus-error@-1:20 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.insbr x10, x20
# plus-error@-1:18 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.insbr x0, x19, x5
# plus-error@-1:10 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.insbr x10, x19, x0
# plus-error@-1:20 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.insbr x10, x19, x5 # minus-error:1 {{instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)}}


qc.insbhr x15, x4, 6
# plus-error@-1:20 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.insbhr x15, x4
# plus-error@-1:18 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.insbhr x0, x4, x6
# plus-error@-1:11 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.insbhr x15, x4, x0
# plus-error@-1:20 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.insbhr x15, x4, x6 # minus-error:1 {{instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)}}


qc.insbpr x21, x8, 9
# plus-error@-1:20 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.insbpr x21, x8
# plus-error@-1:18 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.insbpr x0, x8, x9
# plus-error@-1:11 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.insbpr x21, x8, x0
# plus-error@-1:20 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.insbpr x21, x8, x9 # minus-error:1 {{instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)}}


qc.insbprh x2, x3, 11
# plus-error@-1:20 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.insbprh x2, x3
# plus-error@-1:18 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.insbprh x0, x3, x11
# plus-error@-1:12 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.insbprh x2, x3, x0
# plus-error@-1:20 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.insbprh x2, x3, x11 # minus-error:1 {{instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)}}


qc.extdur x9, x19, 29
# plus-error@-1:20 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.extdur x9, x19
# plus-error@-1:18 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.extdur x0, x19, x29
# plus-error@-1:11 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.extdur x9, x31, x29
# plus-error@-1:15 {{register must be a GPR excluding t6 (x31)}}
# minus-error@-2:1 {{invalid instruction}}

qc.extdur x9, x19, x0
# plus-error@-1:20 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.extdur x9, x19, x29 # minus-error:1 {{instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)}}


qc.extdr x12, x29, 30
# plus-error@-1:20 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.extdr x12, x29
# plus-error@-1:18 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.extdr x0, x29, x30
# plus-error@-1:10 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.extdr x12, x31, x30
# plus-error@-1:15 {{register must be a GPR excluding t6 (x31)}}
# minus-error@-2:1 {{invalid instruction}}

qc.extdr x12, x29, x0
# plus-error@-1:20 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.extdr x12, x29, x30 # minus-error:1 {{instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)}}


qc.extdupr x13, x23, 3
# plus-error@-1:22 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.extdupr x13, x23
# plus-error@-1:20 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.extdupr x0, x23, x3
# plus-error@-1:12 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.extdupr x13, x31, x3
# plus-error@-1:17 {{register must be a GPR excluding t6 (x31)}}
# minus-error@-2:1 {{invalid instruction}}

qc.extdupr x13, x23, x0
# plus-error@-1:22 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.extdupr x13, x23, x3 # minus-error:1 {{instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)}}


qc.extduprh x18, x8, 9
# plus-error@-1:22 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.extduprh x18, x8
# plus-error@-1:20 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.extduprh x0, x8, x9
# plus-error@-1:13 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.extduprh x18, x31, x9
# plus-error@-1:18 {{register must be a GPR excluding t6 (x31)}}
# minus-error@-2:1 {{invalid instruction}}

qc.extduprh x18, x8, x0
# plus-error@-1:22 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.extduprh x18, x8, x9 # minus-error:1 {{instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)}}


qc.extdpr x1, x4, 15
# plus-error@-1:19 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.extdpr x1, x4
# plus-error@-1:17 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.extdpr x0, x4, x15
# plus-error@-1:11 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.extdpr x1, x31, x15
# plus-error@-1:15 {{register must be a GPR excluding t6 (x31)}}
# minus-error@-2:1 {{invalid instruction}}

qc.extdpr x1, x4, x0
# plus-error@-1:19 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.extdpr x1, x4, x15 # minus-error:1 {{instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)}}


qc.extdprh x6, x24, 25
# plus-error@-1:21 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.extdprh x6, x24
# plus-error@-1:19 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.extdprh x0, x24, x25
# plus-error@-1:12 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.extdprh x6, x31, x25
# plus-error@-1:16 {{register must be a GPR excluding t6 (x31)}}
# minus-error@-2:1 {{invalid instruction}}

qc.extdprh x6, x24, x0
# plus-error@-1:21 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.extdprh x6, x24, x25 # minus-error:1 {{instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)}}


qc.c.bexti x1, 8
# plus-error@-1:12 {{register must be a GPR from x8 to x15}}
# minus-error@-2:1 {{invalid instruction}}

qc.c.bexti x15
# plus-error@-1:15 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.c.bexti x15, 43
# plus-error@-1:17 {{immediate must be an integer in the range [1, 31]}}
# minus-error@-2:1 {{invalid instruction}}

qc.c.bexti x9, 8 # minus-error:1 {{instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)}}


qc.c.bseti x2, 10
# plus-error@-1:12 {{register must be a GPR from x8 to x15}}
# minus-error@-2:1 {{invalid instruction}}

qc.c.bseti x12
# plus-error@-1:15 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.c.bseti x12, -10
# plus-error@-1:17 {{immediate must be an integer in the range [1, 31]}}
# minus-error@-2:1 {{invalid instruction}}

qc.c.bseti x12, 30 # minus-error:1 {{instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)}}


qc.c.extu x0, 10
# plus-error@-1:11 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.c.extu x5
# plus-error@-1:13 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.c.extu x17, 3
# plus-error@-1:16 {{immediate must be an integer in the range [6, 32]}}
# minus-error@-2:1 {{invalid instruction}}

qc.c.extu x17, 32 # minus-error:1 {{instruction requires the following: 'Xqcibm' (Qualcomm uC Bit Manipulation Extension)}}
