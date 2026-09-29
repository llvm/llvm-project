@ RUN: llvm-mc -triple=arm < %s | FileCheck %s
 .syntax unified
_foo:

 OBJECT .req r2
 mov r4, OBJECT
 mov r4, oBjEcT
 .unreq oBJECT

_foo2:
 OBJECT .req r5
 mov r4, OBJECT
 .unreq OBJECT

@ CHECK-LABEL: _foo:
@ CHECK: mov r4, r2
@ CHECK: mov r4, r2

@ CHECK-LABEL: _foo2:
@ CHECK: mov r4, r5
