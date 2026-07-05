; RUN: not llc --mtriple=loongarch32 --mattr=+32s,+lasx < %s 2>&1 | FileCheck %s
; RUN: not llc --mtriple=loongarch64 --mattr=+lasx < %s 2>&1 | FileCheck %s

declare <8 x i32> @llvm.loongarch.lasx.xvinsve0.w(<8 x i32>, <8 x i32>, i32)

define <8 x i32> @lasx_xvinsve0_w_lo(<8 x i32> %va, <8 x i32> %vb) nounwind {
; CHECK: llvm.loongarch.lasx.xvinsve0.w: argument out of range
entry:
  %res = call <8 x i32> @llvm.loongarch.lasx.xvinsve0.w(<8 x i32> %va, <8 x i32> %vb, i32 -1)
  ret <8 x i32> %res
}

define <8 x i32> @lasx_xvinsve0_w_hi(<8 x i32> %va, <8 x i32> %vb) nounwind {
; CHECK: llvm.loongarch.lasx.xvinsve0.w: argument out of range
entry:
  %res = call <8 x i32> @llvm.loongarch.lasx.xvinsve0.w(<8 x i32> %va, <8 x i32> %vb, i32 8)
  ret <8 x i32> %res
}

declare <4 x i64> @llvm.loongarch.lasx.xvinsve0.d(<4 x i64>, <4 x i64>, i32)

define <4 x i64> @lasx_xvinsve0_d_lo(<4 x i64> %va, <4 x i64> %vb) nounwind {
; CHECK: llvm.loongarch.lasx.xvinsve0.d: argument out of range
entry:
  %res = call <4 x i64> @llvm.loongarch.lasx.xvinsve0.d(<4 x i64> %va, <4 x i64> %vb, i32 -1)
  ret <4 x i64> %res
}

define <4 x i64> @lasx_xvinsve0_d_hi(<4 x i64> %va, <4 x i64> %vb) nounwind {
; CHECK: llvm.loongarch.lasx.xvinsve0.d: argument out of range
entry:
  %res = call <4 x i64> @llvm.loongarch.lasx.xvinsve0.d(<4 x i64> %va, <4 x i64> %vb, i32 4)
  ret <4 x i64> %res
}
