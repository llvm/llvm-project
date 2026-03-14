@ RUN: not llvm-mc -triple=arm-linux-gnueabi -filetype=obj < %s 2>&1 | FileCheck %s

.text
    cmp r0, #(l1 - unknownLabel + 4) >> 2
@ CHECK: error: expected relocatable expression

l1:
