# RUN: not llvm-mc -triple=riscv32 -mattr=+experimental-xqccmt -M no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-ERROR %s
# RUN: not llvm-mc -triple=riscv64 -mattr=+experimental-xqccmt -M no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-ERROR %s

# CHECK-ERROR: error: immediate must be an integer in the range [0, 31]
qc.cm.jt 32

# CHECK-ERROR: error: immediate must be an integer in the range [32, 255]
qc.cm.jalt 256

# CHECK-ERROR: error: immediate must be an integer in the range [32, 255]
qc.cm.jalt 31
