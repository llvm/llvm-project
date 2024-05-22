; RUN: not llc --mtriple=loongarch64 --mattr=+lasx < %s 2>&1 | FileCheck %s

declare <32 x i8> @llvm.loongarch.lasx.xvslei.b(<32 x i8>, i32)

define <32 x i8> @lasx_xvslei_b_lo(<32 x i8> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvslei.b: argument out of range
entry:
  %res = call <32 x i8> @llvm.loongarch.lasx.xvslei.b(<32 x i8> %va, i32 -17)
  ret <32 x i8> %res
}

define <32 x i8> @lasx_xvslei_b_hi(<32 x i8> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvslei.b: argument out of range
entry:
  %res = call <32 x i8> @llvm.loongarch.lasx.xvslei.b(<32 x i8> %va, i32 16)
  ret <32 x i8> %res
}

declare <16 x i16> @llvm.loongarch.lasx.xvslei.h(<16 x i16>, i32)

define <16 x i16> @lasx_xvslei_h_lo(<16 x i16> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvslei.h: argument out of range
entry:
  %res = call <16 x i16> @llvm.loongarch.lasx.xvslei.h(<16 x i16> %va, i32 -17)
  ret <16 x i16> %res
}

define <16 x i16> @lasx_xvslei_h_hi(<16 x i16> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvslei.h: argument out of range
entry:
  %res = call <16 x i16> @llvm.loongarch.lasx.xvslei.h(<16 x i16> %va, i32 16)
  ret <16 x i16> %res
}

declare <8 x i32> @llvm.loongarch.lasx.xvslei.w(<8 x i32>, i32)

define <8 x i32> @lasx_xvslei_w_lo(<8 x i32> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvslei.w: argument out of range
entry:
  %res = call <8 x i32> @llvm.loongarch.lasx.xvslei.w(<8 x i32> %va, i32 -17)
  ret <8 x i32> %res
}

define <8 x i32> @lasx_xvslei_w_hi(<8 x i32> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvslei.w: argument out of range
entry:
  %res = call <8 x i32> @llvm.loongarch.lasx.xvslei.w(<8 x i32> %va, i32 16)
  ret <8 x i32> %res
}

declare <4 x i64> @llvm.loongarch.lasx.xvslei.d(<4 x i64>, i32)

define <4 x i64> @lasx_xvslei_d_lo(<4 x i64> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvslei.d: argument out of range
entry:
  %res = call <4 x i64> @llvm.loongarch.lasx.xvslei.d(<4 x i64> %va, i32 -17)
  ret <4 x i64> %res
}

define <4 x i64> @lasx_xvslei_d_hi(<4 x i64> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvslei.d: argument out of range
entry:
  %res = call <4 x i64> @llvm.loongarch.lasx.xvslei.d(<4 x i64> %va, i32 16)
  ret <4 x i64> %res
}

declare <32 x i8> @llvm.loongarch.lasx.xvslei.bu(<32 x i8>, i32)

define <32 x i8> @lasx_xvslei_bu_lo(<32 x i8> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvslei.bu: argument out of range
entry:
  %res = call <32 x i8> @llvm.loongarch.lasx.xvslei.bu(<32 x i8> %va, i32 -1)
  ret <32 x i8> %res
}

define <32 x i8> @lasx_xvslei_bu_hi(<32 x i8> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvslei.bu: argument out of range
entry:
  %res = call <32 x i8> @llvm.loongarch.lasx.xvslei.bu(<32 x i8> %va, i32 32)
  ret <32 x i8> %res
}

declare <16 x i16> @llvm.loongarch.lasx.xvslei.hu(<16 x i16>, i32)

define <16 x i16> @lasx_xvslei_hu_lo(<16 x i16> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvslei.hu: argument out of range
entry:
  %res = call <16 x i16> @llvm.loongarch.lasx.xvslei.hu(<16 x i16> %va, i32 -1)
  ret <16 x i16> %res
}

define <16 x i16> @lasx_xvslei_hu_hi(<16 x i16> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvslei.hu: argument out of range
entry:
  %res = call <16 x i16> @llvm.loongarch.lasx.xvslei.hu(<16 x i16> %va, i32 32)
  ret <16 x i16> %res
}

declare <8 x i32> @llvm.loongarch.lasx.xvslei.wu(<8 x i32>, i32)

define <8 x i32> @lasx_xvslei_wu_lo(<8 x i32> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvslei.wu: argument out of range
entry:
  %res = call <8 x i32> @llvm.loongarch.lasx.xvslei.wu(<8 x i32> %va, i32 -1)
  ret <8 x i32> %res
}

define <8 x i32> @lasx_xvslei_wu_hi(<8 x i32> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvslei.wu: argument out of range
entry:
  %res = call <8 x i32> @llvm.loongarch.lasx.xvslei.wu(<8 x i32> %va, i32 32)
  ret <8 x i32> %res
}

declare <4 x i64> @llvm.loongarch.lasx.xvslei.du(<4 x i64>, i32)

define <4 x i64> @lasx_xvslei_du_lo(<4 x i64> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvslei.du: argument out of range
entry:
  %res = call <4 x i64> @llvm.loongarch.lasx.xvslei.du(<4 x i64> %va, i32 -1)
  ret <4 x i64> %res
}

define <4 x i64> @lasx_xvslei_du_hi(<4 x i64> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvslei.du: argument out of range
entry:
  %res = call <4 x i64> @llvm.loongarch.lasx.xvslei.du(<4 x i64> %va, i32 32)
  ret <4 x i64> %res
}
