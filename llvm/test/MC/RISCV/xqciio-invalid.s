# Xqciio - Qualcomm uC External Input Output Extension
# RUN: llvm-mc --verify=expected,plus -triple riscv32 -mattr=+xqciio < %s
# RUN: llvm-mc --verify=expected,minus -triple riscv32 -mattr=-xqciio < %s

# expected-error@+1:18 {{expected register}}
qc.outw x5, 2048(10)

# plus-error@+2:13 {{expected '('}}
# minus-error@+1:1 {{instruction requires the following: 'Xqciio' (Qualcomm uC External Input Output Extension)}}
qc.outw x5, x10

# plus-error@+2:13 {{immediate must be a multiple of 4 bytes in the range [0, 16380]}}
# minus-error@+1:1 {{invalid instruction}}
qc.outw x5, 4099(x10)

# minus-error@+1:1 {{instruction requires the following: 'Xqciio' (Qualcomm uC External Input Output Extension)}}
qc.outw x5, 2048(x10)

# expected-error@+1:19 {{expected register}}
qc.inw x23, 16380(17)

# plus-error@+2:13 {{expected '('}}
# minus-error@+1:1 {{instruction requires the following: 'Xqciio' (Qualcomm uC External Input Output Extension)}}
qc.inw x23, x17

# plus-error@+2:13 {{immediate must be a multiple of 4 bytes in the range [0, 16380]}}
# minus-error@+1:1 {{invalid instruction}}
qc.inw x23, 16384(x17)

# minus-error@+1:1 {{instruction requires the following: 'Xqciio' (Qualcomm uC External Input Output Extension)}}
qc.inw x23, 16380(x17)
