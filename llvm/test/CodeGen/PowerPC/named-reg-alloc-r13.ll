; RUN: not llc < %s -mtriple=powerpc-apple-darwin 2>&1 | FileCheck %s --check-prefix=CHECK-DARWIN
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu 2>&1 | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu 2>&1 | FileCheck %s

define i32 @get_reg() nounwind {
entry:
; FIXME: Include an allocatable-specific error message
; CHECK-DARWIN: Invalid register name global variable
        %reg = call i32 @llvm.read_register.i32(metadata !0)
  ret i32 %reg

; CHECK-LABEL: @get_reg
; CHECK: mr 3, 13
}

declare i32 @llvm.read_register.i32(metadata) nounwind

!0 = !{!"r13\00"}
