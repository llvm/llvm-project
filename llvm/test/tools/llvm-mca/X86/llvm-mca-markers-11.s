# RUN: not llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=btver2 %s 2>&1 | FileCheck %s

# LLVM-MCA-BEGIN foo
add %eax, %eax
# LLVM-MCA-BEGIN foo
add %eax, %eax

# CHECK:      llvm-mca-markers-11.s:5:2: error: overlapping regions cannot have the same name
# CHECK-NEXT: # LLVM-MCA-BEGIN foo
# CHECK-NEXT:  ^
# CHECK-NEXT: llvm-mca-markers-11.s:3:2: note: region foo was previously defined here
# CHECK-NEXT: # LLVM-MCA-BEGIN foo
# CHECK-NEXT:  ^
