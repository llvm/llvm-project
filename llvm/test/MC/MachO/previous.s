// RUN: llvm-mc -triple i386-apple-darwin9 %s -o - | FileCheck %s

.text
// CHECK: .section __TEXT,__text

.data
// CHECK: .section __DATA,__data

.previous
// CHECK: .section __TEXT,__text

.previous
// CHECK: .section __DATA,__data
