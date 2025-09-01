; RUN: not llc --mtriple=loongarch32 --mattr=+32s,+lasx < %s 2>&1 | FileCheck %s
; RUN: not llc --mtriple=loongarch64 --mattr=+lasx < %s 2>&1 | FileCheck %s

declare <8 x i32> @llvm.loongarch.lasx.xvinsgr2vr.w(<8 x i32>, i32, i32)

define <8 x i32> @lasx_xvinsgr2vr_w_lo(<8 x i32> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvinsgr2vr.w: argument out of range
entry:
  %res = call <8 x i32> @llvm.loongarch.lasx.xvinsgr2vr.w(<8 x i32> %va, i32 1, i32 -1)
  ret <8 x i32> %res
}

define <8 x i32> @lasx_xvinsgr2vr_w_hi(<8 x i32> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvinsgr2vr.w: argument out of range
entry:
  %res = call <8 x i32> @llvm.loongarch.lasx.xvinsgr2vr.w(<8 x i32> %va, i32 1, i32 8)
  ret <8 x i32> %res
}

declare <4 x i64> @llvm.loongarch.lasx.xvinsgr2vr.d(<4 x i64>, i64, i32)

define <4 x i64> @lasx_xvinsgr2vr_d_lo(<4 x i64> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvinsgr2vr.d: argument out of range
entry:
  %res = call <4 x i64> @llvm.loongarch.lasx.xvinsgr2vr.d(<4 x i64> %va, i64 1, i32 -1)
  ret <4 x i64> %res
}

define <4 x i64> @lasx_xvinsgr2vr_d_hi(<4 x i64> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvinsgr2vr.d: argument out of range
entry:
  %res = call <4 x i64> @llvm.loongarch.lasx.xvinsgr2vr.d(<4 x i64> %va, i64 1, i32 4)
  ret <4 x i64> %res
}
