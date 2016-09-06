; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple | FileCheck %s

; DAGCombine to transform a conversion of an extract_vector_elt to an
; extract_vector_elt of a conversion, which saves a round trip of copies
; of the value to a GPR and back to and FPR.
; rdar://11855286
define double @foo0(<2 x i64> %a) nounwind {
; CHECK:  scvtf.2d  [[REG:v[0-9]+]], v0, #9
; CHECK-NEXT:  mov  d0, [[REG]][1]
  %vecext = extractelement <2 x i64> %a, i32 1
  %fcvt_n = tail call double @llvm.aarch64.neon.vcvtfxs2fp.f64.i64(i64 %vecext, i32 9)
  ret double %fcvt_n
}

declare double @llvm.aarch64.neon.vcvtfxs2fp.f64.i64(i64, i32) nounwind readnone
