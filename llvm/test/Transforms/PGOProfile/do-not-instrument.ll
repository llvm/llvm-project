; RUN: opt < %s -pgo-instr-gen -S | FileCheck %s
; RUN: opt < %s -passes=pgo-instr-gen -S | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

define i32 @f1() {
; CHECK-LABEL: @f1
entry:
; CHECK: call void @llvm.instrprof.increment
; CHECK-NOT: ptrtoint void (i8*)* asm sideeffect
; CHECK-NOT: call void @llvm.instrprof.value.profile
; CHECK: tail call void asm sideeffect 
  tail call void asm sideeffect "", "imr,~{memory},~{dirflag},~{fpsr},~{flags}"(i8* undef) #0
  ret i32 0
}

define i32 @f2() {
entry:
; CHECK: call void @llvm.instrprof.increment
; CHECK-NOT: call void @llvm.instrprof.value.profile
  call void (i32, ...) bitcast (void (...)* @foo to void (i32, ...)*)(i32 21)
  ret i32 0
}

declare void @foo(...) #0

attributes #0 = { nounwind }
