@ RUN: not llvm-mc -triple=arm-linux-gnueabi -filetype=obj < %s 2>&1 | FileCheck %s

.text
    cmp r0, #(l1 - unknownLabel)
@ CHECK: error: symbol 'unknownLabel' can not be undefined in a subtraction expression

l1:
