; RUN: opt < %s -passes=pgo-instr-gen -S | FileCheck %s

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

define i32 @f1() {
; CHECK-LABEL: @f1
entry:
; CHECK: call void @llvm.instrprof.increment
; CHECK-NOT: ptrtoint ptr asm sideeffect
; CHECK-NOT: call void @llvm.instrprof.value.profile
; CHECK: tail call void asm sideeffect 
  tail call void asm sideeffect "", "imr,~{memory},~{dirflag},~{fpsr},~{flags}"(ptr undef) #0
  ret i32 0
}

define i32 @f2() {
entry:
; CHECK: call void @llvm.instrprof.increment
; CHECK-NOT: call void @llvm.instrprof.value.profile
  call void (i32, ...) @foo(i32 21)
  ret i32 0
}

declare void @foo(...) #0

attributes #0 = { nounwind }
