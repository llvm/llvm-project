# RUN: llvm-mc -triple=x86_64 %s | FileCheck %s

# CHECK:              .globl  inside_at_0
# CHECK-NEXT:         .globl  inside_plus_0
# CHECK-NEXT: inside_at_0:
# CHECK-NEXT: inside_plus_0:
# CHECK-NEXT:         .globl  after_at_0
# CHECK-NEXT:         .globl  after_plus_0
# CHECK-NEXT: after_at_0:
# CHECK-NEXT: after_plus_0:

.macro outer1
  .macro inner1
    .globl inside_at_\@
    .globl inside_plus_\+
    inside_at_\@:
    inside_plus_\+:
  .endm
  inner1
  .globl after_at_\@
  .globl after_plus_\+
  after_at_\@:
  after_plus_\+:
.endm

outer1

# PR18599
.macro macro_a
 .macro macro_b
  .byte 10
  .macro macro_c
  .endm

  macro_c
  .purgem macro_c
 .endm

 macro_b
.endm

# CHECK: .byte 10
# CHECK: .byte 10
macro_a
macro_b
