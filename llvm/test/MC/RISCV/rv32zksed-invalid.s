# With Zks extension:
# RUN: not llvm-mc -triple=riscv32 -mattr=+zks < %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

# With Zksed extension:
# RUN: not llvm-mc -triple=riscv32 -mattr=+zksed < %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

# CHECK-ERROR: immediate must be an integer in the range [0, 3]
sm4ed a0, a1, a2, 8

# CHECK-ERROR: immediate must be an integer in the range [0, 3]
sm4ks a0, a1, a2, 8
