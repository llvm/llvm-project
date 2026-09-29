# With Zk extension:
# RUN: not llvm-mc -triple=riscv64 -mattr=+zk < %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

# With Zkn extension:
# RUN: not llvm-mc -triple=riscv64 -mattr=+zkn < %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

# With Zkne extension:
# RUN: not llvm-mc -triple=riscv64 -mattr=+zkne < %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

# CHECK-ERROR: immediate must be an integer in the range [0, 10]
aes64ks1i a0, a1, 11

# CHECK-ERROR: immediate must be an integer in the range [0, 10]
aes64ks1i a0, a1, -1
