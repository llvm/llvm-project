; RUN: llvm-mc -triple avr -mattr=avr6 -filetype=obj < %s 2>&1 | llvm-objdump -r - | FileCheck %s

; CHECK: R_AVR_CALL foo+0x7fffff
jmp foo+8388607

