; RUN: not llc --mtriple=loongarch32 --mattr=+32s,+lsx < %s 2>&1 | FileCheck %s
; RUN: not llc --mtriple=loongarch64 --mattr=+lsx < %s 2>&1 | FileCheck %s

declare <16 x i8> @llvm.loongarch.lsx.vssrlrni.b.h(<16 x i8>, <16 x i8>, i32)

define <16 x i8> @lsx_vssrlrni_b_h(<16 x i8> %va, <16 x i8> %vb, i32 %c) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <16 x i8> @llvm.loongarch.lsx.vssrlrni.b.h(<16 x i8> %va, <16 x i8> %vb, i32 %c)
  ret <16 x i8> %res
}

declare <8 x i16> @llvm.loongarch.lsx.vssrlrni.h.w(<8 x i16>, <8 x i16>, i32)

define <8 x i16> @lsx_vssrlrni_h_w(<8 x i16> %va, <8 x i16> %vb, i32 %c) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <8 x i16> @llvm.loongarch.lsx.vssrlrni.h.w(<8 x i16> %va, <8 x i16> %vb, i32 %c)
  ret <8 x i16> %res
}

declare <4 x i32> @llvm.loongarch.lsx.vssrlrni.w.d(<4 x i32>, <4 x i32>, i32)

define <4 x i32> @lsx_vssrlrni_w_d(<4 x i32> %va, <4 x i32> %vb, i32 %c) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <4 x i32> @llvm.loongarch.lsx.vssrlrni.w.d(<4 x i32> %va, <4 x i32> %vb, i32 %c)
  ret <4 x i32> %res
}

declare <2 x i64> @llvm.loongarch.lsx.vssrlrni.d.q(<2 x i64>, <2 x i64>, i32)

define <2 x i64> @lsx_vssrlrni_d_q(<2 x i64> %va, <2 x i64> %vb, i32 %c) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <2 x i64> @llvm.loongarch.lsx.vssrlrni.d.q(<2 x i64> %va, <2 x i64> %vb, i32 %c)
  ret <2 x i64> %res
}

declare <16 x i8> @llvm.loongarch.lsx.vssrlrni.bu.h(<16 x i8>, <16 x i8>, i32)

define <16 x i8> @lsx_vssrlrni_bu_h(<16 x i8> %va, <16 x i8> %vb, i32 %c) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <16 x i8> @llvm.loongarch.lsx.vssrlrni.bu.h(<16 x i8> %va, <16 x i8> %vb, i32 %c)
  ret <16 x i8> %res
}

declare <8 x i16> @llvm.loongarch.lsx.vssrlrni.hu.w(<8 x i16>, <8 x i16>, i32)

define <8 x i16> @lsx_vssrlrni_hu_w(<8 x i16> %va, <8 x i16> %vb, i32 %c) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <8 x i16> @llvm.loongarch.lsx.vssrlrni.hu.w(<8 x i16> %va, <8 x i16> %vb, i32 %c)
  ret <8 x i16> %res
}

declare <4 x i32> @llvm.loongarch.lsx.vssrlrni.wu.d(<4 x i32>, <4 x i32>, i32)

define <4 x i32> @lsx_vssrlrni_wu_d(<4 x i32> %va, <4 x i32> %vb, i32 %c) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <4 x i32> @llvm.loongarch.lsx.vssrlrni.wu.d(<4 x i32> %va, <4 x i32> %vb, i32 %c)
  ret <4 x i32> %res
}

declare <2 x i64> @llvm.loongarch.lsx.vssrlrni.du.q(<2 x i64>, <2 x i64>, i32)

define <2 x i64> @lsx_vssrlrni_du_q(<2 x i64> %va, <2 x i64> %vb, i32 %c) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <2 x i64> @llvm.loongarch.lsx.vssrlrni.du.q(<2 x i64> %va, <2 x i64> %vb, i32 %c)
  ret <2 x i64> %res
}
