; RUN: not llc --mtriple=loongarch32 --mattr=+32s,+lasx < %s 2>&1 | FileCheck %s
; RUN: not llc --mtriple=loongarch64 --mattr=+lasx < %s 2>&1 | FileCheck %s

declare <32 x i8> @llvm.loongarch.lasx.xvandi.b(<32 x i8>, i32)

define <32 x i8> @lasx_xvandi_b_lo(<32 x i8> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvandi.b: argument out of range
entry:
  %res = call <32 x i8> @llvm.loongarch.lasx.xvandi.b(<32 x i8> %va, i32 -1)
  ret <32 x i8> %res
}

define <32 x i8> @lasx_xvandi_b_hi(<32 x i8> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvandi.b: argument out of range
entry:
  %res = call <32 x i8> @llvm.loongarch.lasx.xvandi.b(<32 x i8> %va, i32 256)
  ret <32 x i8> %res
}
