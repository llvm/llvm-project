; RUN: not llvm-mc -triple avr -mattr=avr6 -filetype=obj < %s 2>&1 | FileCheck %s

; CHECK: error: out of range port number (expected an integer in the range 0 to 65535)
lds r2, foo+65536

