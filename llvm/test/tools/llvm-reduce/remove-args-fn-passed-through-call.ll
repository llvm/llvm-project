; Test that llvm-reduce can remove uninteresting function arguments from function definitions as well as their calls.
; This test checks that functions with different argument types are handled correctly
;
; RUN: llvm-reduce --abort-on-invalid-reduction --test FileCheck --test-arg --check-prefixes=CHECK-ALL,CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefixes=CHECK-ALL,CHECK-FINAL %s --input-file %t

declare void @pass(ptr)

define void @bar() {
entry:
  ; CHECK-INTERESTINGNESS: call void @pass({{.*}}@interesting
  ; CHECK-FINAL: call void @pass(ptr @interesting)
  call void @pass(ptr @interesting)
  ret void
}

; CHECK-ALL: define internal void @interesting
; CHECK-INTERESTINGNESS-SAME: ({{.*}}%interesting{{.*}}) {
; CHECK-FINAL-SAME: (ptr %interesting)
define internal void @interesting(i32 %uninteresting1, ptr %uninteresting2, ptr %interesting) {
entry:
  ret void
}
