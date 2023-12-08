; RUN: not llc --mtriple=loongarch64 --mattr=+lsx < %s 2>&1 | FileCheck %s

declare <16 x i8> @llvm.loongarch.lsx.vld(i8*, i32)

define <16 x i8> @lsx_vld_lo(i8* %p) nounwind {
; CHECK: llvm.loongarch.lsx.vld: argument out of range
entry:
  %res = call <16 x i8> @llvm.loongarch.lsx.vld(i8* %p, i32 -2049)
  ret <16 x i8> %res
}

define <16 x i8> @lsx_vld_hi(i8* %p) nounwind {
; CHECK: llvm.loongarch.lsx.vld: argument out of range
entry:
  %res = call <16 x i8> @llvm.loongarch.lsx.vld(i8* %p, i32 2048)
  ret <16 x i8> %res
}
