; Test that llvm-reduce can remove dso_local.
;
; RUN: llvm-reduce --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefix=CHECK-FINAL %s < %t

; CHECK-INTERESTINGNESS: declare
; CHECK-INTERESTINGNESS-SAME: void @f0
; CHECK-INTERESTINGNESS-SAME: i32
; CHECK-INTERESTINGNESS-SAME: i32

; CHECK-FINAL: declare void @f0(i32, i32)

declare dso_local void @f0(i32, i32)

; CHECK-INTERESTINGNESS: declare
; CHECK-INTERESTINGNESS-SAME: dso_local
; CHECK-INTERESTINGNESS-SAME: void @f1
; CHECK-INTERESTINGNESS-SAME: i32
; CHECK-INTERESTINGNESS-SAME: i32

; CHECK-FINAL: declare dso_local void @f1(i32, i32)

declare dso_local void @f1(i32, i32)

