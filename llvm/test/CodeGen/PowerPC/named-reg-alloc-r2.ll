; RUN: not --crash llc < %s -mtriple=powerpc-unknown-linux-gnu 2>&1 | FileCheck %s
; RUN: not --crash llc < %s -mtriple=powerpc64-unknown-linux-gnu 2>&1 | FileCheck %s --check-prefix=CHECK-NOTPPC32

define i32 @get_reg() nounwind {
entry:
; CHECK-NOTPPC32: Trying to reserve an invalid register "r2".
; CHECK: Trying to obtain a reserved register "r2".
        %reg = call i32 @llvm.read_register.i32(metadata !0)
  ret i32 %reg
}

declare i32 @llvm.read_register.i32(metadata) nounwind

!0 = !{!"r2\00"}
