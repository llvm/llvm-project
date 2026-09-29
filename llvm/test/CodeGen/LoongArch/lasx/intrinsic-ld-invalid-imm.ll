; RUN: not llc --mtriple=loongarch32 --mattr=+32s,+lasx < %s 2>&1 | FileCheck %s
; RUN: not llc --mtriple=loongarch64 --mattr=+lasx < %s 2>&1 | FileCheck %s

declare <32 x i8> @llvm.loongarch.lasx.xvld(ptr, i32)

define <32 x i8> @lasx_xvld_lo(ptr %p) nounwind {
; CHECK: llvm.loongarch.lasx.xvld: argument out of range
entry:
  %res = call <32 x i8> @llvm.loongarch.lasx.xvld(ptr %p, i32 -2049)
  ret <32 x i8> %res
}

define <32 x i8> @lasx_xvld_hi(ptr %p) nounwind {
; CHECK: llvm.loongarch.lasx.xvld: argument out of range
entry:
  %res = call <32 x i8> @llvm.loongarch.lasx.xvld(ptr %p, i32 2048)
  ret <32 x i8> %res
}
