# RUN: not llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=btver2 %s 2>&1 | FileCheck %s

# LLVM-MCA-BEGIN
# LLVM-MCA-END

# LLVM-MCA-BEGIN
# LLVM-MCA-END

# CHECK: error: no assembly instructions found.
