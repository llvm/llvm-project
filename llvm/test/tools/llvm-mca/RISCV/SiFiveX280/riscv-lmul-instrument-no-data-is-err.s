# RUN: not llvm-mca -mtriple=riscv64 -mcpu=sifive-x280 -iterations=1 < %s 2>&1 | FileCheck %s

vsetvli zero, a0, e8, m1, tu, mu
# LLVM-MCA-RISCV-LMUL
vadd.vv v12, v12, v12

# CHECK: error: Failed to create RISCV-LMUL instrument with no data
# CHECK: # LLVM-MCA-RISCV-LMUL
# CHECK:  ^
# CHECK: error: There was an error parsing comments.
