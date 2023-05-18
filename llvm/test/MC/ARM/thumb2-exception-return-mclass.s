# RUN: not llvm-mc -triple thumbv7m -assemble < %s 2>&1 | FileCheck %s

  .text

# CHECK: instruction requires: !armv*m
# CHECK-NEXT: srsdb sp, #7
  srsdb sp, #7

# CHECK: instruction requires: !armv*m
# CHECK-NEXT: rfeia r6
  rfeia r6

# CHECK: instruction requires: !armv*m
# CHECK-NEXT: subs pc, lr, #42
  subs pc, lr, #42
