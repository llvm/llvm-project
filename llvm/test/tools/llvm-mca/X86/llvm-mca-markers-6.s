# RUN: not llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=btver2 %s 2>&1 | FileCheck %s

# LLVM-MCA-BEGIN  foo

# LLVM-MCA-BEGIN  bar

# LLVM-MCA-END

# CHECK:      llvm-mca-markers-6.s:7:2: error: found an invalid region end directive
# CHECK-NEXT: # LLVM-MCA-END
# CHECK-NEXT:  ^
# CHECK-NEXT: llvm-mca-markers-6.s:7:2: note: unable to find an active anonymous region
# CHECK-NEXT: # LLVM-MCA-END
# CHECK-NEXT:  ^
