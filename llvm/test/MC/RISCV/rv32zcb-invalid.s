# RUN: not llvm-mc -triple=riscv32 -mattr=experimental-zcb -riscv-no-aliases -show-encoding %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-ERROR %s
# RUN: not llvm-mc -triple=riscv64 -mattr=experimental-zcb -riscv-no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-ERROR %s

# CHECK-ERROR: error: immediate must be an integer in the range [0, 3]
c.lbu a5, 10(a4)

# CHECK-ERROR: error: immediate must be one of [0, 2]
c.lhu a5, 10(a4)

# CHECK-ERROR: error: immediate must be one of [0, 2]
c.lh a5, 10(a4)

# CHECK-ERROR: error: immediate must be an integer in the range [0, 3]
c.sb a5, 10(a4)

# CHECK-ERROR: error: immediate must be one of [0, 2]
c.sh a5, 10(a4)
