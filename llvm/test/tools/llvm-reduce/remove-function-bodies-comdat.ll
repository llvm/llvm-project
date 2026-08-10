; RUN: llvm-reduce --abort-on-invalid-reduction --test FileCheck --test-arg --check-prefixes=CHECK-ALL,CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: cat %t | FileCheck --check-prefixes=CHECK-ALL,CHECK-FINAL %s
; RUN: opt -passes=verify %t

; CHECK-FINAL-NOT: = comdat
; CHECK-INTERESTINGNESS: @callee(
; CHECK-FINAL: declare void @callee()

$foo = comdat any

define void @callee() comdat($foo) {
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
