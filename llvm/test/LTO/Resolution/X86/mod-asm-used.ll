; RUN: opt -module-summary -o %t.bc %s
; RUN: opt -module-summary -o %t2.bc %S/Inputs/mod-asm-used.ll
; RUN: llvm-lto2 run %t.bc -r %t.bc,foo,l %t2.bc -r %t2.bc,foo,pl -o %t3 -save-temps
; RUN: llvm-nm %t3.? | FileCheck %s

; RUN: llvm-dis %t3.index.bc -o - | FileCheck %s --check-prefix=INDEX
; INDEX: ^0 = module: (path: "{{.*}}mod-asm-used.ll.tmp.bc"
; INDEX: ^1 = module: (path: "{{.*}}mod-asm-used.ll.tmp2.bc"

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: D foo
module asm ".quad foo"
