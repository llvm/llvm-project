; RUN: llc -mtriple=agx -mcpu=agx-g13g -verify-machineinstrs %s -o - | FileCheck %s
;
; Matrix expansion introduces lane-dependent shuffle indices. Analyze the
; expanded instructions before deciding which shuffles require quad repair.
; The simulator verifier checks every output lane against an independent sum.
; CHECK-LABEL: matrix_lane_inputs:
; CHECK: simd_shuffle
; CHECK: fmadd32

declare i32 @llvm.agx.thread.index.in.simdgroup()
declare <2 x float> @llvm.agx.simdgroup.matrix.fma.v2f32(<2 x float>, <2 x float>, <2 x float>)

define agx_kernel void @matrix_lane_inputs(ptr addrspace(1) %output) {
  %lane = call i32 @llvm.agx.thread.index.in.simdgroup()
  %f = uitofp i32 %lane to float
  %twice = fmul float %f, 2.0
  %a0 = fadd float %twice, 1.0
  %a1 = fadd float %twice, 2.0
  %a_first = insertelement <2 x float> poison, float %a0, i32 0
  %a = insertelement <2 x float> %a_first, float %a1, i32 1
  %result = call <2 x float> @llvm.agx.simdgroup.matrix.fma.v2f32(
      <2 x float> %a, <2 x float> <float 1.0, float 1.0>, <2 x float> zeroinitializer)
  %at = getelementptr <2 x float>, ptr addrspace(1) %output, i32 %lane
  store <2 x float> %result, ptr addrspace(1) %at
  ret void
}
