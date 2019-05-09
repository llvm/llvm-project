# RUN: not llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=btver2 %s 2>&1 | FileCheck %s

# LLVM-MCA-BEGIN
add %eax, %eax
# LLVM-MCA-BEGIN
add %eax, %eax

# CHECK:      llvm-mca-markers-12.s:5:2: error: found multiple overlapping anonymous regions
# CHECK-NEXT: # LLVM-MCA-BEGIN
# CHECK-NEXT:  ^
# CHECK-NEXT: llvm-mca-markers-12.s:3:2: note: Previous anonymous region was defined here
# CHECK-NEXT: # LLVM-MCA-BEGIN
# CHECK-NEXT:  ^
