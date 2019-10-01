// RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux %s -o -| llvm-readobj --symbols | FileCheck %s


.globl f1
.type f1, @function
.set micromips
f1:
	nop

.globl d1
.type d1, @object
d1:
.word 42

.globl f2
.type f2, @function
.set nomicromips
f2:
	nop

// CHECK-LABEL: Name: d1
// CHECK:       Other: 0
// CHECK-LABEL: Name: f1
// CHECK:       Other [ (0x80)
// CHECK-LABEL: Name: f2
// CHECK:       Other: 0
