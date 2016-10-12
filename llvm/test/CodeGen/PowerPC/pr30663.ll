; RUN: llc -O1 < %s | FileCheck %s
target triple = "powerpc64le-linux-gnu"

; The second xxspltw should be eliminated.
; CHECK: xxspltw
; CHECK-NOT: xxspltw
define void @Test() {
bb4:
  %tmp = load <4 x i8>, <4 x i8>* undef
  %tmp8 = bitcast <4 x i8> %tmp to float
  %tmp18 = fmul float %tmp8, undef
  %tmp19 = fsub float 0.000000e+00, %tmp18
  store float %tmp19, float* undef
  %tmp22 = shufflevector <4 x i8> %tmp, <4 x i8> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3>
  %tmp23 = bitcast <16 x i8> %tmp22 to <4 x float>
  %tmp25 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> undef, <4 x float> %tmp23, <4 x float> undef)
  %tmp26 = fsub <4 x float> zeroinitializer, %tmp25
  %tmp27 = bitcast <4 x float> %tmp26 to <4 x i32>
  tail call void @llvm.ppc.altivec.stvx(<4 x i32> %tmp27, i8* undef)
  ret void
}

declare void @llvm.ppc.altivec.stvx(<4 x i32>, i8*)
declare <4 x float> @llvm.fma.v4f32(<4 x float>, <4 x float>, <4 x float>)
