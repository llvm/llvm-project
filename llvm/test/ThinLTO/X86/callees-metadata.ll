; Do setup work: generate bitcode and combined index
; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/callees-metadata.ll -o %t2.bc

; RUN: llvm-lto2 run %t1.bc %t2.bc -o %t.o -save-temps \
; RUN:     -r=%t1.bc,bar,plx \
; RUN:     -r=%t1.bc,foo,l \
; RUN:     -r=%t2.bc,foo,pl
; RUN: llvm-dis %t.o.1.3.import.bc -o - | FileCheck %s
; CHECK: define {{.*}} i32 @f1.llvm.0
; CHECK: define {{.*}} i32 @f2.llvm.0

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local i32 @bar(i32 %x) {
entry:
  %call = call i32 @foo(i32 %x)
  ret i32 %call
}

declare dso_local i32 @foo(i32)
