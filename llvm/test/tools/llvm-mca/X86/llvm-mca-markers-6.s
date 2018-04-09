# RUN: not llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=btver2 %s 2>&1 | FileCheck %s

# LLVM-MCA-BEGIN  foo

# LLVM-MCA-BEGIN  bar

# LLVM-MCA-END

# CHECK:      llvm-mca-markers-6.s:5:2: warning: Ignoring invalid region start
# CHECK-NEXT: # LLVM-MCA-BEGIN  bar
# CHECK-NEXT:  ^
# CHECK-NEXT: error: no assembly instructions found.

