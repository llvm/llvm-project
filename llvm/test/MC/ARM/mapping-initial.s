@ RUN: llvm-mc -triple=armv7a-none-eabi     -filetype=obj < %s | llvm-objdump -t - | FileCheck %s --check-prefix=ARM
@ RUN: llvm-mc -triple=armebv7a-none-eabi   -filetype=obj < %s | llvm-objdump -t - | FileCheck %s --check-prefix=ARM
@ RUN: llvm-mc -triple=thumbv7a-none-eabi   -filetype=obj < %s | llvm-objdump -t - | FileCheck %s --check-prefix=THUMB
@ RUN: llvm-mc -triple=thumbebv7a-none-eabi -filetype=obj < %s | llvm-objdump -t - | FileCheck %s --check-prefix=THUMB

  add r0, r0, r0

@ ARM:      00000000         .text  00000000 $a
@ THUMB:    00000000         .text  00000000 $t
