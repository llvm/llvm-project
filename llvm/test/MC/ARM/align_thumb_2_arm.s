@ RUN: llvm-mc -triple thumbv7-none-linux -filetype=obj -o %t.o %s
@ RUN: llvm-objdump --triple=armv7-none-linux -d %t.o | FileCheck --check-prefix=THUMB_2_ARM %s

@ RUN: llvm-mc -triple thumbv7-apple-darwin -filetype=obj -o %t_darwin.o %s
@ RUN: llvm-objdump --triple=armv7-apple-darwin -d %t_darwin.o | FileCheck --check-prefix=THUMB_2_ARM %s

.syntax unified
.code 32
@ THUMB_2_ARM-LABEL: foo
foo:
  add r0, r0
.align 3
@ THUMB_2_ARM: 4: 00 f0 20 e3    nop
  add r0, r0

