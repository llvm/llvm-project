; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=CHECK %s

; CHECK-LABEL: {{^}}test1:
; CHECK: v_cndmask_b32_e64 v0, 0, 1, exec
;
; Note: We could generate better code here if we recognized earlier that
; there is no WQM use and therefore llvm.amdgcn.ps.live is constant. However,
; the expectation is that the intrinsic will be used in non-trivial shaders,
; so such an optimization doesn't seem worth the effort.
define amdgpu_ps float @test1() {
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
define amdgpu_ps float @test2() {
  %live = call i1 @llvm.amdgcn.ps.live()
  %live.32 = zext i1 %live to i32

  %t = call <4 x float> @llvm.SI.image.sample.i32(i32 %live.32, <8 x i32> undef, <4 x i32> undef, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)

  %r = extractelement <4 x float> %t, i32 0
  ret float %r
}

; CHECK-LABEL: {{^}}test3:
; CHECK: s_mov_b64 [[LIVE:s\[[0-9]+:[0-9]+\]]], exec
; CHECK-DAG: s_wqm_b64 exec, exec
; CHECK-DAG: s_xor_b64 [[HELPER:s\[[0-9]+:[0-9]+\]]], [[LIVE]], -1
; CHECK_DAG: s_and_saveexec_b64 [[SAVED:s\[[0-9]+:[0-9]+\]]], [[HELPER]]
; CHECK: ; %dead
define amdgpu_ps float @test3(i32 %in) {
entry:
  %live = call i1 @llvm.amdgcn.ps.live()
  br i1 %live, label %end, label %dead

dead:
  %tc.dead = mul i32 %in, 2
  br label %end

end:
  %tc = phi i32 [ %in, %entry ], [ %tc.dead, %dead ]
  %t = call <4 x float> @llvm.SI.image.sample.i32(i32 %tc, <8 x i32> undef, <4 x i32> undef, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)

  %r = extractelement <4 x float> %t, i32 0
  ret float %r
}

declare i1 @llvm.amdgcn.ps.live() #0

declare <4 x float> @llvm.SI.image.sample.i32(i32, <8 x i32>, <4 x i32>, i32, i32, i32, i32, i32, i32, i32, i32) #0

attributes #0 = { nounwind readnone }
