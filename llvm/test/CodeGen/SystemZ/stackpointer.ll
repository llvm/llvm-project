; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

define ptr @get_stack() nounwind {
entry:
; CHECK-LABEL: get_stack:
; CHECK: lgr %r2, %r15
; CHECK-NEXT: br %r14
        %0 = call i64 @llvm.read_register.i64(metadata !0)
        %1 = inttoptr i64 %0 to ptr
  ret ptr %1
}

declare i64 @llvm.read_register.i64(metadata) nounwind

!0 = !{!"r15"}
