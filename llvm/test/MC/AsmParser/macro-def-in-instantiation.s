// RUN: llvm-mc -triple x86_64-apple-darwin10 %s | FileCheck %s

.macro .make_macro
$0 $1
$2 $3
$4
.endmacro

.make_macro .macro,.mybyte,.byte,$0,.endmacro

.data
// CHECK: .byte 10
.mybyte 10
