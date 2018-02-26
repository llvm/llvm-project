; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple -mcpu=cyclone | FileCheck %s
; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple -O0 -fast-isel -mcpu=cyclone | FileCheck %s --check-prefix=CHECK-FAST

define <16 x i8> @foo(<16 x i8> %a) nounwind optsize readnone ssp {
; CHECK: uaddlv.16b h0, v0
; CHECK: rshrn.8b v0, v0, #4
; CHECK: dup.16b v0, v0[0]
; CHECK: ret

; CHECK-FAST: uaddlv.16b
; CHECK-FAST: rshrn.8b
; CHECK-FAST: dup.16b
  %tmp = tail call i32 @llvm.aarch64.neon.uaddlv.i32.v16i8(<16 x i8> %a) nounwind
  %tmp1 = trunc i32 %tmp to i16
  %tmp2 = insertelement <8 x i16> undef, i16 %tmp1, i32 0
  %tmp3 = tail call <8 x i8> @llvm.aarch64.neon.rshrn.v8i8(<8 x i16> %tmp2, i32 4)
  %tmp4 = shufflevector <8 x i8> %tmp3, <8 x i8> undef, <16 x i32> zeroinitializer
  ret <16 x i8> %tmp4
}

declare <8 x i8> @llvm.aarch64.neon.rshrn.v8i8(<8 x i16>, i32) nounwind readnone
declare i32 @llvm.aarch64.neon.uaddlv.i32.v16i8(<16 x i8>) nounwind readnone
