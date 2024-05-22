; RUN: not llc --mtriple=loongarch64 --mattr=+lasx < %s 2>&1 | FileCheck %s

declare <32 x i8> @llvm.loongarch.lasx.xvbitseli.b(<32 x i8>, <32 x i8>, i32)

define <32 x i8> @lasx_xvbitseli_b_lo(<32 x i8> %va, <32 x i8> %vb) nounwind {
; CHECK: llvm.loongarch.lasx.xvbitseli.b: argument out of range
entry:
  %res = call <32 x i8> @llvm.loongarch.lasx.xvbitseli.b(<32 x i8> %va, <32 x i8> %vb, i32 -1)
  ret <32 x i8> %res
}

define <32 x i8> @lasx_xvbitseli_b_hi(<32 x i8> %va, <32 x i8> %vb) nounwind {
; CHECK: llvm.loongarch.lasx.xvbitseli.b: argument out of range
entry:
  %res = call <32 x i8> @llvm.loongarch.lasx.xvbitseli.b(<32 x i8> %va, <32 x i8> %vb, i32 256)
  ret <32 x i8> %res
}
