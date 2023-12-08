; RUN: llc < %s -mtriple=s390x-ibm-zos | FileCheck %s

; CHECK-LABEL: get_stack:
; CHECK: lgr   3, 4
; CHECK: b 2(7)

define ptr @get_stack() nounwind {
entry:
        %0 = call i64 @llvm.read_register.i64(metadata !0)
        %1 = inttoptr i64 %0 to ptr
  ret ptr %1
}

declare i64 @llvm.read_register.i64(metadata) nounwind

!0 = !{!"r4"}
