; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=CHECK %s

; CHECK-LABEL: {{^}}test1:
; CHECK: v_cndmask_b32_e64 v0, 0, 1, exec
;
; Note: We could generate better code here if we recognized earlier that
; there is no WQM use and therefore llvm.amdgcn.ps.live is constant. However,
; the expectation is that the intrinsic will be used in non-trivial shaders,
; so such an optimization doesn't seem worth the effort.
define amdgpu_ps float @test1() #0 {
  %live = call i1 @llvm.amdgcn.ps.live()
  %live.32 = zext i1 %live to i32
  %r = bitcast i32 %live.32 to float
  ret float %r
}

; CHECK-LABEL: {{^}}test2:
; CHECK: s_mov_b64 [[LIVE:s\[[0-9]+:[0-9]+\]]], exec
; CHECK-DAG: s_wqm_b64 exec, exec
; CHECK-DAG: v_cndmask_b32_e64 [[VAR:v[0-9]+]], 0, 1, [[LIVE]]
; CHECK: image_sample v0, [[VAR]],
define amdgpu_ps float @test2() #0 {
  %live = call i1 @llvm.amdgcn.ps.live()
  %live.32 = zext i1 %live to i32
  %live.32.bc = bitcast i32 %live.32 to float
  %t = call <4 x float> @llvm.amdgcn.image.sample.v4f32.f32.v8i32(float %live.32.bc, <8 x i32> undef, <4 x i32> undef, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %r = extractelement <4 x float> %t, i32 0
  ret float %r
}

; CHECK-LABEL: {{^}}test3:
; CHECK: s_mov_b64 [[LIVE:s\[[0-9]+:[0-9]+\]]], exec
; CHECK-DAG: s_wqm_b64 exec, exec
; CHECK-DAG: s_xor_b64 [[HELPER:s\[[0-9]+:[0-9]+\]]], [[LIVE]], -1
; CHECK_DAG: s_and_saveexec_b64 [[SAVED:s\[[0-9]+:[0-9]+\]]], [[HELPER]]
; CHECK: ; %dead
define amdgpu_ps float @test3(i32 %in) #0 {
entry:
  %live = call i1 @llvm.amdgcn.ps.live()
  br i1 %live, label %end, label %dead

dead:
  %tc.dead = mul i32 %in, 2
  br label %end

end:
  %tc = phi i32 [ %in, %entry ], [ %tc.dead, %dead ]
  %tc.bc = bitcast i32 %tc to float
  %t = call <4 x float> @llvm.amdgcn.image.sample.v4f32.f32.v8i32(float %tc.bc, <8 x i32> undef, <4 x i32> undef, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false) #0
  %r = extractelement <4 x float> %t, i32 0
  ret float %r
}

declare i1 @llvm.amdgcn.ps.live() #1
declare <4 x float> @llvm.amdgcn.image.sample.v4f32.f32.v8i32(float, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #2

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind readonly }
