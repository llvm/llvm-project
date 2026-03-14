@ RUN: llvm-mc -triple=arm-linux-gnueabi < %s | FileCheck %s

@ CHECK: ldr	r12, [sp, #15]
ldr r12, [sp, (15)]

@ CHECK: ldr	r12, [sp, #15]
ldr r12, [sp, #(15)]

@ CHECK: ldr	r12, [sp, #15]
ldr r12, [sp, $(15)]

@ CHECK: ldr	r12, [sp, #100]
ldr r12, [sp, (((15+5)*5))]

@ CHECK: ldr	r12, [sp, #100]
ldr r12, [sp, #(((15+5)*5))]


@ CHECK: ldr	r12, [sp, #100]
ldr r12, [sp, $(((15+5)*5))]
