; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/callees-metadata-malformed.ll -o %t2.bc
; RUN: llvm-lto2 run %t1.bc %t2.bc -o %t.o -save-temps \
; RUN:     -r=%t1.bc,caller,plx \
; RUN:     -r=%t1.bc,callee,l \
; RUN:     -r=%t2.bc,callee,pl
; RUN: llvm-dis %t.o.1.3.import.bc -o - | FileCheck %s

; A mixed malformed !callees node must not add an edge for its valid-looking
; operand to the module summary.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: define available_externally void @callee
; CHECK-NOT: define {{.*}} @metadata_only_target

define void @caller(ptr %target) {
  call void @callee(ptr %target)
  ret void
}

declare void @callee(ptr)
