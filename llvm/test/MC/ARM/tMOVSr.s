@ REQUIRES: asserts
@ RUN: llvm-mc --triple=thumbv8 %s --show-encoding 2>&1 | FileCheck %s --match-full-lines

// Note this makes sure the narrow instruciton is selected
@ CHECK: movs r2, r3 @ encoding: [0x1a,0x00]
.text
  movs r2, r3
