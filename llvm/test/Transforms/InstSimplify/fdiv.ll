; RUN: opt < %s -instsimplify -S | FileCheck %s

define double @fdiv_of_undef(double %X) {
; CHECK: @fdiv_of_undef
; undef / X -> undef
  %r = fdiv double undef, %X
  ret double %r
; CHECK: ret double undef
}

define double @fdiv_by_undef(double %X) {
; CHECK: @fdiv_by_undef
; X / undef -> undef
  %r = fdiv double %X, undef
  ret double %r
; CHECK: ret double undef
}
