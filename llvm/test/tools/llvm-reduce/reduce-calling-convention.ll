; Test that llvm-reduce can remove function calling conventions.
;
; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=function-data --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefix=CHECK-FINAL --implicit-check-not=fastcc %s < %t

; CHECK-INTERESTINGNESS: declare
; CHECK-INTERESTINGNESS-SAME: void @f

; CHECK-FINAL: declare void @f()

declare void @f()

; CHECK-INTERESTINGNESS: declare
; CHECK-INTERESTINGNESS-SAME: void @g

; CHECK-FINAL: declare void @g()

declare fastcc void @g()
