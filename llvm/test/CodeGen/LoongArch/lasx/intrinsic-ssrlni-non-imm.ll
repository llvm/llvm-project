; RUN: not llc --mtriple=loongarch32 --mattr=+32s,+lasx < %s 2>&1 | FileCheck %s
; RUN: not llc --mtriple=loongarch64 --mattr=+lasx < %s 2>&1 | FileCheck %s

declare <32 x i8> @llvm.loongarch.lasx.xvssrlni.b.h(<32 x i8>, <32 x i8>, i32)

define <32 x i8> @lasx_xvssrlni_b_h(<32 x i8> %va, <32 x i8> %vb, i32 %c) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <32 x i8> @llvm.loongarch.lasx.xvssrlni.b.h(<32 x i8> %va, <32 x i8> %vb, i32 %c)
  ret <32 x i8> %res
}

declare <16 x i16> @llvm.loongarch.lasx.xvssrlni.h.w(<16 x i16>, <16 x i16>, i32)

define <16 x i16> @lasx_xvssrlni_h_w(<16 x i16> %va, <16 x i16> %vb, i32 %c) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <16 x i16> @llvm.loongarch.lasx.xvssrlni.h.w(<16 x i16> %va, <16 x i16> %vb, i32 %c)
  ret <16 x i16> %res
}

declare <8 x i32> @llvm.loongarch.lasx.xvssrlni.w.d(<8 x i32>, <8 x i32>, i32)

define <8 x i32> @lasx_xvssrlni_w_d(<8 x i32> %va, <8 x i32> %vb, i32 %c) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <8 x i32> @llvm.loongarch.lasx.xvssrlni.w.d(<8 x i32> %va, <8 x i32> %vb, i32 %c)
  ret <8 x i32> %res
}

declare <4 x i64> @llvm.loongarch.lasx.xvssrlni.d.q(<4 x i64>, <4 x i64>, i32)

define <4 x i64> @lasx_xvssrlni_d_q(<4 x i64> %va, <4 x i64> %vb, i32 %c) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <4 x i64> @llvm.loongarch.lasx.xvssrlni.d.q(<4 x i64> %va, <4 x i64> %vb, i32 %c)
  ret <4 x i64> %res
}

declare <32 x i8> @llvm.loongarch.lasx.xvssrlni.bu.h(<32 x i8>, <32 x i8>, i32)

define <32 x i8> @lasx_xvssrlni_bu_h(<32 x i8> %va, <32 x i8> %vb, i32 %c) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <32 x i8> @llvm.loongarch.lasx.xvssrlni.bu.h(<32 x i8> %va, <32 x i8> %vb, i32 %c)
  ret <32 x i8> %res
}

declare <16 x i16> @llvm.loongarch.lasx.xvssrlni.hu.w(<16 x i16>, <16 x i16>, i32)

define <16 x i16> @lasx_xvssrlni_hu_w(<16 x i16> %va, <16 x i16> %vb, i32 %c) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <16 x i16> @llvm.loongarch.lasx.xvssrlni.hu.w(<16 x i16> %va, <16 x i16> %vb, i32 %c)
  ret <16 x i16> %res
}

declare <8 x i32> @llvm.loongarch.lasx.xvssrlni.wu.d(<8 x i32>, <8 x i32>, i32)

define <8 x i32> @lasx_xvssrlni_wu_d(<8 x i32> %va, <8 x i32> %vb, i32 %c) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <8 x i32> @llvm.loongarch.lasx.xvssrlni.wu.d(<8 x i32> %va, <8 x i32> %vb, i32 %c)
  ret <8 x i32> %res
}

declare <4 x i64> @llvm.loongarch.lasx.xvssrlni.du.q(<4 x i64>, <4 x i64>, i32)

define <4 x i64> @lasx_xvssrlni_du_q(<4 x i64> %va, <4 x i64> %vb, i32 %c) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <4 x i64> @llvm.loongarch.lasx.xvssrlni.du.q(<4 x i64> %va, <4 x i64> %vb, i32 %c)
  ret <4 x i64> %res
}
