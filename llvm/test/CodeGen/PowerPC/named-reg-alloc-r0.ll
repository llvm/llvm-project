; RUN: not llc < %s -mtriple=powerpc-unknown-linux-gnu 2>&1 | FileCheck %s
; RUN: not llc < %s -mtriple=powerpc-unknown-linux-gnu 2>&1 | FileCheck %s
; RUN: not llc < %s -mtriple=powerpc64-unknown-linux-gnu 2>&1 | FileCheck %s

define i32 @get_reg() nounwind {
entry:
; FIXME: Include an allocatable-specific error message
; CHECK: Invalid register name global variable
        %reg = call i32 @llvm.read_register.i32(metadata !0)
  ret i32 %reg
}

declare i32 @llvm.read_register.i32(metadata) nounwind

!0 = !{!"r0\00"}
