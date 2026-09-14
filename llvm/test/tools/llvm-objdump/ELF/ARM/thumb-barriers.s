@ RUN: llvm-mc -triple=thumbv7m-none-eabi -filetype=obj %s -o %t.o
@ RUN: llvm-objdump -D --arch-name=thumb %t.o | FileCheck %s
@ RUN: llvm-objdump -D %t.o | FileCheck %s

@ Check that overriding the architecture with generic Thumb still decodes
@ barriers with all their operands, even without the data-barrier feature.

.text
.arch armv7-m
.thumb
foo:
  isb sy
  dsb sy
  dmb sy

@ CHECK-LABEL: <foo>:
@ CHECK-NEXT: 0: f3bf 8f6f isb sy
@ CHECK-NEXT: 4: f3bf 8f4f dsb sy
@ CHECK-NEXT: 8: f3bf 8f5f dmb sy
