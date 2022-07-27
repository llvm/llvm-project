; RUN: llc < %s -mtriple=i686-- -mcpu=corei7 -mattr=+avx2 | FileCheck %s

declare x86_fastcallcc i64 @barrier()

;CHECK-LABEL: bcast_fold:
;CHECK: vmov{{[au]}}ps %xmm{{[0-9]+}}, [[SPILLED:[^\)]+\)]]
;CHECK: barrier
;CHECK: vbroadcastss [[SPILLED]], %ymm0
;CHECK: ret
define <8 x float> @bcast_fold( ptr %A) {
BB:
  %A0 = load float, ptr %A
  %tt3 = call x86_fastcallcc i64 @barrier()
  br i1 undef, label %work, label %exit

work:
  %A1 = insertelement <8 x float> undef, float %A0, i32 0
  %A2 = shufflevector <8 x float> %A1, <8 x float> undef, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x float> %A2

exit:
  ret <8 x float> undef
}
