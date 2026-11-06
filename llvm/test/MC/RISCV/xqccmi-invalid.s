# Xqccmi - Qualcomm 16-bit Instruction Lookup Table
# RUN: llvm-mc --verify -triple=riscv32 -mattr=+experimental-xqccmi -M no-aliases -show-encoding < %s

# expected-error@+1 {{immediate must be an integer in the range [0, 2047]}}
qc.cm.ilut 2048

# expected-error@+1 {{immediate must be an integer in the range [0, 2047]}}
qc.cm.ilut -1
