; RUN: not llc --mtriple=loongarch64 --mattr=+lasx < %s 2>&1 | FileCheck %s

declare <32 x i8> @llvm.loongarch.lasx.xvsubi.bu(<32 x i8>, i32)

define <32 x i8> @lasx_xvsubi_bu_lo(<32 x i8> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvsubi.bu: argument out of range
entry:
  %res = call <32 x i8> @llvm.loongarch.lasx.xvsubi.bu(<32 x i8> %va, i32 -1)
  ret <32 x i8> %res
}

define <32 x i8> @lasx_xvsubi_bu_hi(<32 x i8> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvsubi.bu: argument out of range
entry:
  %res = call <32 x i8> @llvm.loongarch.lasx.xvsubi.bu(<32 x i8> %va, i32 32)
  ret <32 x i8> %res
}

declare <16 x i16> @llvm.loongarch.lasx.xvsubi.hu(<16 x i16>, i32)

define <16 x i16> @lasx_xvsubi_hu_lo(<16 x i16> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvsubi.hu: argument out of range
entry:
  %res = call <16 x i16> @llvm.loongarch.lasx.xvsubi.hu(<16 x i16> %va, i32 -1)
  ret <16 x i16> %res
}

define <16 x i16> @lasx_xvsubi_hu_hi(<16 x i16> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvsubi.hu: argument out of range
entry:
  %res = call <16 x i16> @llvm.loongarch.lasx.xvsubi.hu(<16 x i16> %va, i32 32)
  ret <16 x i16> %res
}

declare <8 x i32> @llvm.loongarch.lasx.xvsubi.wu(<8 x i32>, i32)

define <8 x i32> @lasx_xvsubi_wu_lo(<8 x i32> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvsubi.wu: argument out of range
entry:
  %res = call <8 x i32> @llvm.loongarch.lasx.xvsubi.wu(<8 x i32> %va, i32 -1)
  ret <8 x i32> %res
}

define <8 x i32> @lasx_xvsubi_wu_hi(<8 x i32> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvsubi.wu: argument out of range
entry:
  %res = call <8 x i32> @llvm.loongarch.lasx.xvsubi.wu(<8 x i32> %va, i32 32)
  ret <8 x i32> %res
}

declare <4 x i64> @llvm.loongarch.lasx.xvsubi.du(<4 x i64>, i32)

define <4 x i64> @lasx_xvsubi_du_lo(<4 x i64> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvsubi.du: argument out of range
entry:
  %res = call <4 x i64> @llvm.loongarch.lasx.xvsubi.du(<4 x i64> %va, i32 -1)
  ret <4 x i64> %res
}

define <4 x i64> @lasx_xvsubi_du_hi(<4 x i64> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvsubi.du: argument out of range
entry:
  %res = call <4 x i64> @llvm.loongarch.lasx.xvsubi.du(<4 x i64> %va, i32 32)
  ret <4 x i64> %res
}
