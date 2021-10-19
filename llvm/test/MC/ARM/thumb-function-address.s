@ RUN: llvm-mc -filetype=obj -triple=armv7-linux-gnueabi %s -o %t
@ RUN: llvm-readelf -s %t | FileCheck %s

@@ GNU as sets the thumb state according to the thumb state of the label. If a
@@ .type directive is placed after the label, set the symbol's thumb state
@@ according to the thumb state of the .type directive. This matches GNU as in
@@ most cases.

.syntax unified
.text
.thumb
func_label:
.type func_label, %function

.type foo_impl, %function
foo_impl:
  bx lr
.type foo_resolver, %function
foo_resolver:
  b foo_impl
.type foo, %gnu_indirect_function
.set foo, foo_resolver

@@ Note: GNU as sets the value to 1.
.thumb
label:
  bx lr
.arm
  bx lr
.type label, %function

@@ Check func_label, foo_impl, foo_resolver, and foo addresses have bit 0 set.
@@ Check label has bit 0 unset.
@ CHECK:      Value Size Type   Bind  Vis     Ndx Name
@ CHECK-NEXT: 00000000 0 NOTYPE LOCAL DEFAULT     UND
@ CHECK-NEXT: 00000001 0 FUNC   LOCAL DEFAULT 2   func_label
@ CHECK-NEXT: 00000001 0 FUNC   LOCAL DEFAULT 2   foo_impl
@ CHECK-NEXT: 00000000 0 NOTYPE LOCAL DEFAULT 2   $t.0
@ CHECK-NEXT: 00000003 0 FUNC   LOCAL DEFAULT 2   foo_resolver
@ CHECK-NEXT: 00000003 0 IFUNC  LOCAL DEFAULT 2   foo
@ CHECK-NEXT: 00000004 0 FUNC   LOCAL DEFAULT 2   label
@ CHECK-NEXT: 00000006 0 NOTYPE LOCAL DEFAULT 2   $a.1
