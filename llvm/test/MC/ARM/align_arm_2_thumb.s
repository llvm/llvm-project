@ RUN: llvm-mc -triple armv7-none-linux -filetype=obj -o %t.o %s
@ RUN: llvm-objdump --triple=thumbv7-none-linux -d %t.o | FileCheck --check-prefix=ARM_2_THUMB %s

@ RUN: llvm-mc -triple armv7-apple-darwin -filetype=obj -o %t_darwin.o %s
@ RUN: llvm-objdump --triple=thumbv7-apple-darwin -d %t_darwin.o | FileCheck --check-prefix=ARM_2_THUMB %s

.syntax unified
.code 16
@ ARM_2_THUMB-LABEL: foo
foo:
  add r0, r0
.align 3
@ ARM_2_THUMB: 2: bf00      nop
  add r0, r0

