; RUN: llvm-mc -triple avr -mattr=avr6 -filetype=obj < %s | llvm-objdump -r - | FileCheck %s

; CHECK: R_AVR_6_ADIW foo+0x3f
adiw r24, foo+63

