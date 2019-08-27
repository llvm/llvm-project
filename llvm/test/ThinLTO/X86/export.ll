; Do setup work for all below tests: generate bitcode and combined index
; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/export.ll -o %t2.bc
; RUN: llvm-lto -thinlto-action=thinlink -o %t3.bc %t1.bc %t2.bc

; Ensure statics are promoted/renamed correctly from this file.
; RUN: llvm-lto -thinlto-action=promote %t1.bc -thinlto-index=%t3.bc -o - | llvm-dis -o - | FileCheck %s
; CHECK-DAG: @staticvar.llvm.0 = hidden global
; CHECK-DAG: define hidden void @staticfunc.llvm.0

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

@staticvar = internal global i32 1, align 4

define void @callstaticfunc() #0 {
entry:
  call void @staticfunc()
  ret void
}

define internal void @staticfunc() #0 {
entry:
  %0 = load i32, i32* @staticvar, align 4
  ret void
}
