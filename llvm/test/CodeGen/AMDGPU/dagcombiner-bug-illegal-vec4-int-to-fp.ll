;RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

; This test is for a bug in
; DAGCombiner::reduceBuildVecConvertToConvertBuildVec() where
; the wrong type was being passed to
; TargetLowering::getOperationAction() when checking the legality of
; ISD::UINT_TO_FP and ISD::SINT_TO_FP opcodes.


; CHECK: {{^}}sint:
; CHECK: INT_TO_FLT * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

define amdgpu_kernel void @sint(<4 x float> addrspace(1)* %out, i32 addrspace(1)* %in) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %in, i32 1
  %sint = load i32, i32 addrspace(1) * %in
  %conv = sitofp i32 %sint to float
  %0 = insertelement <4 x float> undef, float %conv, i32 0
  %splat = shufflevector <4 x float> %0, <4 x float> undef, <4 x i32> zeroinitializer
  store <4 x float> %splat, <4 x float> addrspace(1)* %out
  ret void
}

;CHECK: {{^}}uint:
;CHECK: UINT_TO_FLT * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

define amdgpu_kernel void @uint(<4 x float> addrspace(1)* %out, i32 addrspace(1)* %in) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %in, i32 1
  %uint = load i32, i32 addrspace(1) * %in
  %conv = uitofp i32 %uint to float
  %0 = insertelement <4 x float> undef, float %conv, i32 0
  %splat = shufflevector <4 x float> %0, <4 x float> undef, <4 x i32> zeroinitializer
  store <4 x float> %splat, <4 x float> addrspace(1)* %out
  ret void
}
