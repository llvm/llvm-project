; RUN: not llc --mtriple=loongarch32 --mattr=+32s,+lasx < %s 2>&1 | FileCheck %s
; RUN: not llc --mtriple=loongarch64 --mattr=+lasx < %s 2>&1 | FileCheck %s

declare <32 x i8> @llvm.loongarch.lasx.xvsrarni.b.h(<32 x i8>, <32 x i8>, i32)

define <32 x i8> @lasx_xvsrarni_b_h(<32 x i8> %va, <32 x i8> %vb, i32 %c) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <32 x i8> @llvm.loongarch.lasx.xvsrarni.b.h(<32 x i8> %va, <32 x i8> %vb, i32 %c)
  ret <32 x i8> %res
}

declare <16 x i16> @llvm.loongarch.lasx.xvsrarni.h.w(<16 x i16>, <16 x i16>, i32)

define <16 x i16> @lasx_xvsrarni_h_w(<16 x i16> %va, <16 x i16> %vb, i32 %c) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <16 x i16> @llvm.loongarch.lasx.xvsrarni.h.w(<16 x i16> %va, <16 x i16> %vb, i32 %c)
  ret <16 x i16> %res
}

declare <8 x i32> @llvm.loongarch.lasx.xvsrarni.w.d(<8 x i32>, <8 x i32>, i32)

define <8 x i32> @lasx_xvsrarni_w_d(<8 x i32> %va, <8 x i32> %vb, i32 %c) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <8 x i32> @llvm.loongarch.lasx.xvsrarni.w.d(<8 x i32> %va, <8 x i32> %vb, i32 %c)
  ret <8 x i32> %res
}

declare <4 x i64> @llvm.loongarch.lasx.xvsrarni.d.q(<4 x i64>, <4 x i64>, i32)

define <4 x i64> @lasx_xvsrarni_d_q(<4 x i64> %va, <4 x i64> %vb, i32 %c) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <4 x i64> @llvm.loongarch.lasx.xvsrarni.d.q(<4 x i64> %va, <4 x i64> %vb, i32 %c)
  ret <4 x i64> %res
}
