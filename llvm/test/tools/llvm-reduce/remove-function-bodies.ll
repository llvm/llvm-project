; RUN: llvm-reduce --abort-on-invalid-reduction --test FileCheck --test-arg --check-prefixes=CHECK-ALL,CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefixes=CHECK-ALL,CHECK-FINAL %s < %t

; CHECK-INTERESTINGNESS: @callee(
; CHECK-FINAL: declare void @callee()
define void @callee() {
  ret void
}

; CHECK-ALL: define void @caller()
define void @caller() {
entry:
; CHECK-ALL: call void @callee()
; CHECK-ALL: ret void
  call void @callee()
  ret void
}
