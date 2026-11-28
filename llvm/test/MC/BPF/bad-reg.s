# RUN: not llvm-mc -triple bpfel < %s 2>&1 | FileCheck %s

# r12 is the last register.
  r13 = 1
# CHECK: [[#@LINE-1]]:3: error: invalid register/token name
  w13 = 1
# CHECK: [[#@LINE-1]]:3: error: invalid register/token name
