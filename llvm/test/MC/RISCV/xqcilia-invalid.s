# Xqcilia - Qualcomm uC Large Immediate Arithmetic extension
# RUN: llvm-mc --verify=plus -triple riscv32 -mattr=+xqcilia < %s
# RUN: llvm-mc --verify=minus -triple riscv32 -mattr=-xqcilia < %s

# plus-error@+2:12 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.e.addai 9, 33554432

# plus-error@+2:14 {{too few operands for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.e.addai x9

# plus-error@+2:16 {{immediate must be an integer in the range [-2147483648, 4294967295]}}
# minus-error@+1 {{invalid instruction}}
qc.e.addai x9, 20485546494

# minus-error@+1:1 {{instruction requires the following: 'Xqcilia' (Qualcomm uC Large Immediate Arithmetic Extension)}}
qc.e.addai x9, 33554432


# plus-error@+2:16 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.e.addi x10, 9, 554432

# plus-error@+2:18 {{too few operands for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.e.addi x10, x9

# plus-error@+2:20 {{immediate must be an integer in the range [-33554432, 33554431]}}
# minus-error@+1 {{invalid instruction}}
qc.e.addi x10, x9, 335544312

# minus-error@+1:1 {{instruction requires the following: 'Xqcilia' (Qualcomm uC Large Immediate Arithmetic Extension)}}
qc.e.addi x10, x9, 554432


# plus-error@+2:12 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.e.andai 9, 33554432

# plus-error@+2:14 {{too few operands for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.e.andai x9

# plus-error@+2:16 {{immediate must be an integer in the range [-2147483648, 4294967295]}}
# minus-error@+1 {{invalid instruction}}
qc.e.andai x9, 20494437494

# minus-error@+1:1 {{instruction requires the following: 'Xqcilia' (Qualcomm uC Large Immediate Arithmetic Extension)}}
qc.e.andai x9, 33554432


# plus-error@+2:16 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.e.andi x10, 9, 554432

# plus-error@+2:18 {{too few operands for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.e.andi x10, x9

# plus-error@+2:20 {{immediate must be an integer in the range [-33554432, 33554431]}}
# minus-error@+1 {{invalid instruction}}
qc.e.andi x10, x9, 335544312

# minus-error@+1:1 {{instruction requires the following: 'Xqcilia' (Qualcomm uC Large Immediate Arithmetic Extension)}}
qc.e.andi x10, x9, 554432


# plus-error@+2:11 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.e.orai 9, 33554432

# plus-error@+2:13 {{too few operands for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.e.orai x9

# plus-error@+2:15 {{immediate must be an integer in the range [-2147483648, 4294967295]}}
# minus-error@+1 {{invalid instruction}}
qc.e.orai x9, 20494437494

# minus-error@+1:1 {{instruction requires the following: 'Xqcilia' (Qualcomm uC Large Immediate Arithmetic Extension)}}
qc.e.orai x9, 33554432


# plus-error@+2:15 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.e.ori x10, 9, 554432

# plus-error@+2:17 {{too few operands for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.e.ori x10, x9

# plus-error@+2:19 {{immediate must be an integer in the range [-33554432, 33554431]}}
# minus-error@+1 {{invalid instruction}}
qc.e.ori x10, x9, 335544312

# minus-error@+1:1 {{instruction requires the following: 'Xqcilia' (Qualcomm uC Large Immediate Arithmetic Extension)}}
qc.e.ori x10, x9, 554432



# plus-error@+2:12 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.e.xorai 9, 33554432

# plus-error@+2:14 {{too few operands for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.e.xorai x9

# plus-error@+2:16 {{immediate must be an integer in the range [-2147483648, 4294967295]}}
# minus-error@+1 {{invalid instruction}}
qc.e.xorai x9, 20494437494

# minus-error@+1:1 {{instruction requires the following: 'Xqcilia' (Qualcomm uC Large Immediate Arithmetic Extension)}}
qc.e.xorai x9, 33554432


# plus-error@+2:16 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.e.xori x10, 9, 554432

# plus-error@+2:18 {{too few operands for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.e.xori x10, x9

# plus-error@+2:20 {{immediate must be an integer in the range [-33554432, 33554431]}}
# minus-error@+1 {{invalid instruction}}
qc.e.xori x10, x9, 335544312

# minus-error@+1:1 {{instruction requires the following: 'Xqcilia' (Qualcomm uC Large Immediate Arithmetic Extension)}}
qc.e.xori x10, x9, 554432
