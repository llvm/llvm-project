; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-apple-darwin 2>&1 | FileCheck %s --check-prefix=CHECK-DARWIN
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu 2>&1 | FileCheck %s

define i64 @get_reg() nounwind {
entry:
  %reg = call i64 @llvm.read_register.i64(metadata !0)
  ret i64 %reg

; CHECK-LABEL: @get_reg
; CHECK: mr 3, 13

; CHECK-DARWIN-LABEL: @get_reg
; CHECK-DARWIN: mr r3, r13
}

declare i64 @llvm.read_register.i64(metadata) nounwind

!0 = !{!"r13\00"}
