# Xqccmi - Qualcomm 16-bit Instruction Lookup Table
# RUN: not llvm-mc -triple=riscv32 -mattr=+experimental-xqccmi -M no-aliases -show-encoding < %s 2>&1 | FileCheck -check-prefix=CHECK-ERROR %s

# CHECK-ERROR: error: immediate must be an integer in the range [0, 2047]
qc.cm.ilut 2048

# CHECK-ERROR: error: immediate must be an integer in the range [0, 2047]
qc.cm.ilut -1
