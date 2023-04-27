# RUN: not llvm-mca -mtriple=riscv64 -mcpu=sifive-x280 -iterations=1 < %s 2>&1 | FileCheck %s

vsetvli zero, a0, e8, m1, tu, mu
# LLVM-MCA-RISCV-V MF9
vadd.vv v12, v12, v12

# CHECK: error: Unknown instrumentation type in LLVM-MCA comment: RISCV-V
# CHECK: # LLVM-MCA-RISCV-V MF9
# CHECK:  ^
# CHECK:  error: There was an error parsing comments.
