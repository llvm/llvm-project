; RUN: not llc --mtriple=loongarch32 --mattr=+32s,+lasx < %s 2>&1 | FileCheck %s
; RUN: not llc --mtriple=loongarch64 --mattr=+lasx < %s 2>&1 | FileCheck %s

declare <32 x i8> @llvm.loongarch.lasx.xvextrins.b(<32 x i8>, <32 x i8>, i32)

define <32 x i8> @lasx_xvextrins_b_lo(<32 x i8> %va, <32 x i8> %vb) nounwind {
; CHECK: llvm.loongarch.lasx.xvextrins.b: argument out of range
entry:
  %res = call <32 x i8> @llvm.loongarch.lasx.xvextrins.b(<32 x i8> %va, <32 x i8> %vb, i32 -1)
  ret <32 x i8> %res
}

define <32 x i8> @lasx_xvextrins_b_hi(<32 x i8> %va, <32 x i8> %vb) nounwind {
; CHECK: llvm.loongarch.lasx.xvextrins.b: argument out of range
entry:
  %res = call <32 x i8> @llvm.loongarch.lasx.xvextrins.b(<32 x i8> %va, <32 x i8> %vb, i32 256)
  ret <32 x i8> %res
}

declare <16 x i16> @llvm.loongarch.lasx.xvextrins.h(<16 x i16>, <16 x i16>, i32)

define <16 x i16> @lasx_xvextrins_h_lo(<16 x i16> %va, <16 x i16> %vb) nounwind {
; CHECK: llvm.loongarch.lasx.xvextrins.h: argument out of range
entry:
  %res = call <16 x i16> @llvm.loongarch.lasx.xvextrins.h(<16 x i16> %va, <16 x i16> %vb, i32 -1)
  ret <16 x i16> %res
}

define <16 x i16> @lasx_xvextrins_h_hi(<16 x i16> %va, <16 x i16> %vb) nounwind {
; CHECK: llvm.loongarch.lasx.xvextrins.h: argument out of range
entry:
  %res = call <16 x i16> @llvm.loongarch.lasx.xvextrins.h(<16 x i16> %va, <16 x i16> %vb, i32 256)
  ret <16 x i16> %res
}

declare <8 x i32> @llvm.loongarch.lasx.xvextrins.w(<8 x i32>, <8 x i32>, i32)

define <8 x i32> @lasx_xvextrins_w_lo(<8 x i32> %va, <8 x i32> %vb) nounwind {
; CHECK: llvm.loongarch.lasx.xvextrins.w: argument out of range
entry:
  %res = call <8 x i32> @llvm.loongarch.lasx.xvextrins.w(<8 x i32> %va, <8 x i32> %vb, i32 -1)
  ret <8 x i32> %res
}

define <8 x i32> @lasx_xvextrins_w_hi(<8 x i32> %va, <8 x i32> %vb) nounwind {
; CHECK: llvm.loongarch.lasx.xvextrins.w: argument out of range
entry:
  %res = call <8 x i32> @llvm.loongarch.lasx.xvextrins.w(<8 x i32> %va, <8 x i32> %vb, i32 256)
  ret <8 x i32> %res
}

declare <4 x i64> @llvm.loongarch.lasx.xvextrins.d(<4 x i64>, <4 x i64>, i32)

define <4 x i64> @lasx_xvextrins_d_lo(<4 x i64> %va, <4 x i64> %vb) nounwind {
; CHECK: llvm.loongarch.lasx.xvextrins.d: argument out of range
entry:
  %res = call <4 x i64> @llvm.loongarch.lasx.xvextrins.d(<4 x i64> %va, <4 x i64> %vb, i32 -1)
  ret <4 x i64> %res
}

define <4 x i64> @lasx_xvextrins_d_hi(<4 x i64> %va, <4 x i64> %vb) nounwind {
; CHECK: llvm.loongarch.lasx.xvextrins.d: argument out of range
entry:
  %res = call <4 x i64> @llvm.loongarch.lasx.xvextrins.d(<4 x i64> %va, <4 x i64> %vb, i32 256)
  ret <4 x i64> %res
}
