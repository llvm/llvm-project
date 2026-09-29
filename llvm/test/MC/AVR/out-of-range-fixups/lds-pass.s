; RUN: llvm-mc -triple avr -mattr=avr6 -filetype=obj < %s | llvm-objdump -r - | FileCheck %s

; CHECK: R_AVR_16 foo+0xffff
lds r2, foo+65535

