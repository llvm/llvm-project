; RUN: llc < %s -mtriple=x86_64-linux | FileCheck %s

; This is a case where we would incorrectly conclude that LBB0_1 could only
; be reached via fall through and would therefore omit the label.

; CHECK:      jne     .LBB0_1
; CHECK-NEXT: jnp     .LBB0_3
; CHECK-NEXT: .LBB0_1:

define void @xyz() {
entry:
  br i1 fcmp oeq (double fsub (double undef, double undef), double 0.000000e+00), label %bar, label %foo

foo:
  br i1 fcmp ogt (double fdiv (double fsub (double fmul (double undef, double undef), double fsub (double undef, double undef)), double fmul (double undef, double undef)), double 1.0), label %foo, label %bar

bar:
  ret void
}
