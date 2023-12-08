@ PR18931
@ RUN: llvm-mc < %s -triple=arm-linux-gnueabi -filetype=obj -o - \
@ RUN: | llvm-objdump -d --arch=arm - | FileCheck %s

    .text
@ CHECK: cmp r2, #1
    cmp r2, #(l2 - l1 + 4) >> 2
l1:
l2:
