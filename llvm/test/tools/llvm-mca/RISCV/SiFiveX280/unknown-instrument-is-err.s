# RUN: not llvm-mca -mtriple=riscv64 -mcpu=sifive-x280 -timeline -iterations=1 < %s 2>&1 | FileCheck %s

# LLVM-MCA-UNKNOWN M1
vsetvli zero, a0, e8, m1, tu, mu
vadd.vv v12, v12, v12

# CHECK: error: Unknown instrumentation type in LLVM-MCA comment: UNKNOWN
# CHECK: # LLVM-MCA-UNKNOWN M1
# CHECK:  ^
# CHECK:  error: There was an error parsing comments.
