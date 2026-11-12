# RUN: not llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=btver2 %s 2>&1 | FileCheck %s

# LLVM-MCA-END foo

# CHECK:      llvm-mca-markers-8.s:3:2: error: found an invalid region end directive
# CHECK-NEXT: # LLVM-MCA-END foo
# CHECK-NEXT:  ^
# CHECK-NEXT: llvm-mca-markers-8.s:3:2: note: unable to find an active region named foo
# CHECK-NEXT: # LLVM-MCA-END foo
# CHECK-NEXT:  ^
