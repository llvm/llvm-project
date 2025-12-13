; RUN: not llc --mtriple=loongarch32 --mattr=+32s,+lasx < %s 2>&1 | FileCheck %s
; RUN: not llc --mtriple=loongarch64 --mattr=+lasx < %s 2>&1 | FileCheck %s

declare <32 x i8> @llvm.loongarch.lasx.xvbsrl.v(<32 x i8>, i32)

define <32 x i8> @lasx_xvbsrl_v_lo(<32 x i8> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvbsrl.v: argument out of range
entry:
  %res = call <32 x i8> @llvm.loongarch.lasx.xvbsrl.v(<32 x i8> %va, i32 -1)
  ret <32 x i8> %res
}

define <32 x i8> @lasx_xvbsrl_v_hi(<32 x i8> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvbsrl.v: argument out of range
entry:
  %res = call <32 x i8> @llvm.loongarch.lasx.xvbsrl.v(<32 x i8> %va, i32 32)
  ret <32 x i8> %res
}
