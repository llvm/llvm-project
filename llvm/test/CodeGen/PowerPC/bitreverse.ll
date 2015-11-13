; RUN: llc -march=ppc64 %s -o - | FileCheck %s

; These tests just check that the plumbing is in place for @llvm.bitreverse. The
; actual output is massive at the moment as llvm.bitreverse is not yet legal.

declare <2 x i16> @llvm.bitreverse.v2i16(<2 x i16>) readnone

define <2 x i16> @f(<2 x i16> %a) {
; CHECK-LABEL: f:
; CHECK: rlwinm
  %b = call <2 x i16> @llvm.bitreverse.v2i16(<2 x i16> %a)
  ret <2 x i16> %b
}

declare i8 @llvm.bitreverse.i8(i8) readnone

define i8 @g(i8 %a) {
; CHECK-LABEL: g:
; CHECK: rlwinm
; CHECK: rlwimi
  %b = call i8 @llvm.bitreverse.i8(i8 %a)
  ret i8 %b
}
