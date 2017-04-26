; RUN: opt -module-summary -o %t.bc %s
; RUN: opt -module-summary -o %t2.bc %S/Inputs/mod-asm-used.ll
; RUN: llvm-lto2 run %t.bc -r %t.bc,foo,l %t2.bc -r %t2.bc,foo,pl -o %t3
; RUN: llvm-nm %t3.1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: D foo
module asm ".quad foo"
