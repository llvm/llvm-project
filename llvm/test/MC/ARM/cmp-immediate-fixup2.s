@ PR18931
@ RUN: llvm-mc < %s -triple=arm-linux-gnueabi -filetype=obj -o - \
@ RUN: | llvm-objdump -d --arch=arm - | FileCheck %s

    .text
@ CHECK: cmp r2, #0
    cmp r2, #(l2 - l1)
l1:
l2:
