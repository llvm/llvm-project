; Tests COFF-specific directives in module level assembly.

; RUN: llc -filetype=obj %s -o %t.obj
; RUN: llvm-readobj -t %t.obj | FileCheck %s
; RUN: opt -thinlto-bc %s -o %t.thinlto.bc
; RUN: llvm-lto2 run %t.thinlto.bc -o %t.thinlto.obj -r=%t.thinlto.bc,foo,plx
; RUN: llvm-readobj -t %t.thinlto.obj.1 | FileCheck %s

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.24210"

module asm ".text"
module asm ".def foo; .scl 3; .type 32; .endef"
module asm ".global foo"
module asm "foo:"
module asm "  ret"

; CHECK: Symbol {
; CHECK:   Name: foo
; CHECK:   StorageClass:
; CHECK-SAME: Static (0x3)
