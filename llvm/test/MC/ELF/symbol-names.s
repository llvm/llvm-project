// RUN: llvm-mc -triple i686-pc-linux -filetype=obj %s -o - | llvm-readobj --symbols | FileCheck %s

// MC allows ?'s in symbol names as an extension.

.text
.globl foo?bar
.type foo?bar, @function
foo?bar:
ret

// CHECK: Symbol
// CHECK: Name: foo?bar
