; Test that llvm-reduce can remove uninteresting function arguments from function definitions as well as their calls.
; This test checks that functions with different argument types are handled correctly
;
; RUN: llvm-reduce --test FileCheck --test-arg --check-prefixes=CHECK-ALL,CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefixes=CHECK-ALL,CHECK-FINAL %s --input-file %t

declare void @pass(void (i32, i8*, i64*)*)

define void @bar() {
entry:
  ; CHECK-INTERESTINGNESS: call void @pass({{.*}}@interesting
  ; CHECK-FINAL: call void @pass(void (i32, i8*, i64*)* bitcast (void (i64*)* @interesting to void (i32, i8*, i64*)*))
  call void @pass(void (i32, i8*, i64*)* @interesting)
  ret void
}

; CHECK-ALL: define internal void @interesting
; CHECK-INTERESTINGNESS-SAME: ({{.*}}%interesting{{.*}}) {
; CHECK-FINAL-SAME: (i64* %interesting)
define internal void @interesting(i32 %uninteresting1, i8* %uninteresting2, i64* %interesting) {
entry:
  ret void
}
