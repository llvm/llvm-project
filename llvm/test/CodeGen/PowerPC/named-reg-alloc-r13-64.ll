; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu 2>&1 | FileCheck %s

define i64 @get_reg() nounwind {
entry:
  %reg = call i64 @llvm.read_register.i64(metadata !0)
  ret i64 %reg

; CHECK-LABEL: @get_reg
; CHECK: mr 3, 13

}

declare i64 @llvm.read_register.i64(metadata) nounwind

!0 = !{!"r13\00"}
