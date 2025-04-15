; RUN: llc < %s -mtriple=ve -mattr=+vpu | FileCheck %s

; Function Attrs: norecurse nounwind readnone
; Check that a single-element insertion is lowered to a insert_vector_elt node for isel.
define fastcc <256 x i32> @expand_single_elem_build_vec(i32 %x, i32 %y) {
; CHECK-LABEL: expand_single_elem_build_vec:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lsv %v0(42), %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %r = insertelement <256 x i32> undef, i32 %x, i32 42
  ret <256 x i32> %r
}
