# RUN: not llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=btver2 %s 2>&1 | FileCheck %s

# LLVM-MCA-BEGIN  foo

# LLVM-MCA-END

# LLVM-MCA-END

# CHECK:      llvm-mca-markers-7.s:7:2: warning: Ignoring invalid region end
# CHECK-NEXT: # LLVM-MCA-END
# CHECK-NEXT:  ^

