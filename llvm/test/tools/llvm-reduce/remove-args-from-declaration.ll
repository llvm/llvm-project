; RUN: llvm-reduce --test FileCheck --test-arg --check-prefixes=CHECK-ALL,CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: cat %t | FileCheck --check-prefixes=CHECK-ALL,CHECK-FINAL %s

; CHECK-INTERESTINGNESS-LABEL: @interesting(
; CHECK-INTERESTINGNESS-SAME: i32
; CHECK-FINAL: declare void @interesting(i32)
declare void @interesting(i32 %uninteresting1, i32 %interesting, i32 %uninteresting2)

; CHECK-INTERESTINGNESS-LABEL: @interesting2(
; CHECK-INTERESTINGNESS-SAME: i32
; CHECK-FINAL: declare void @interesting2(i32)
declare void @interesting2(i32 %uninteresting1, i32 %interesting, i32 %uninteresting2)

; CHECK-INTERESTINGNESS-LABEL: @callee(
; CHECK-INTERESTINGNESS-SAME: i32 %interesting
; CHECK-FINAL: define void @callee(i32 %interesting) {
define void @callee(i32 %uninteresting1, i32 %interesting, i32 %uninteresting2) {
; CHECK-INTERESTINGNESS: call void @interesting2(
; CHECK-INTERESTINGNESS-SAME: i32 %interesting
; CHECK-FINAL: call void @interesting2(i32 %interesting)
  call void @interesting2(i32 %uninteresting1, i32 %interesting, i32 %uninteresting2)
; CHECK-ALL: ret void
  ret void
}
