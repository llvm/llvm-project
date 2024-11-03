; Test that llvm-reduce can remove function linkage.
;
; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=global-values --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefix=CHECK-FINAL %s < %t

; CHECK-INTERESTINGNESS: define
; CHECK-INTERESTINGNESS-SAME: void @f

; CHECK-FINAL: define void @f()

define void @f() {
  ret void
}

; CHECK-INTERESTINGNESS: define
; CHECK-INTERESTINGNESS-SAME: void @g

; CHECK-FINAL: define void @g()

define internal void @g() {
  ret void
}
