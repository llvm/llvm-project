; Test that llvm-reduce can remove uninteresting function arguments from function definitions as well as their calls.
; This test checks that functions with different argument types are handled correctly
;
; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=arguments --test FileCheck --test-arg --check-prefixes=CHECK-ALL,CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefixes=CHECK-ALL,CHECK-FINAL %s --input-file %t

%struct.foo = type { ptr, i32, i32, ptr }

define dso_local void @bar() {
entry:
  ; CHECK-INTERESTINGNESS: call void @interesting(
  ; CHECK-FINAL: call void @interesting(ptr null)
  call void @interesting(i32 0, ptr null, ptr null, ptr null, i64 0)
  ret void
}

; CHECK-ALL: define internal void @interesting
; CHECK-INTERESTINGNESS-SAME: ({{.*}}%interesting{{.*}}) {
; CHECK-FINAL-SAME: (ptr %interesting) {
define internal void @interesting(i32 %uninteresting1, ptr %uninteresting2, ptr %interesting, ptr %uninteresting3, i64 %uninteresting4) {
entry:
  ret void
}
