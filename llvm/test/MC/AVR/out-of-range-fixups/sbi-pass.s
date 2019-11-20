; RUN: llvm-mc -triple avr -mattr=avr6 -filetype=obj < %s | llvm-objdump -r - | FileCheck %s

; CHECK: R_AVR_PORT5 foo+0x1f
sbi foo+31, 1

