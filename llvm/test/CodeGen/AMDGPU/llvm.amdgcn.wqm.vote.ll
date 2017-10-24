; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=CHECK %s

;CHECK-LABEL: {{^}}ret:
;CHECK: v_cmp_eq_u32_e32 [[CMP:[^,]+]], v0, v1
;CHECK: s_wqm_b64 [[WQM:[^,]+]], [[CMP]]
;CHECK: v_cndmask_b32_e64 v0, 0, 1.0, [[WQM]]
define amdgpu_ps float @ret(i32 %v0, i32 %v1) #1 {
main_body:
  %c = icmp eq i32 %v0, %v1
  %w = call i1 @llvm.amdgcn.wqm.vote(i1 %c)
  %r = select i1 %w, float 1.0, float 0.0
  ret float %r
}

;CHECK-LABEL: {{^}}true:
;CHECK: s_wqm_b64
define amdgpu_ps float @true() #1 {
main_body:
  %w = call i1 @llvm.amdgcn.wqm.vote(i1 true)
  %r = select i1 %w, float 1.0, float 0.0
  ret float %r
}

;CHECK-LABEL: {{^}}false:
;CHECK: s_wqm_b64
define amdgpu_ps float @false() #1 {
main_body:
  %w = call i1 @llvm.amdgcn.wqm.vote(i1 false)
  %r = select i1 %w, float 1.0, float 0.0
  ret float %r
}

;CHECK-LABEL: {{^}}kill:
;CHECK: v_cmp_eq_u32_e32 [[CMP:[^,]+]], v0, v1
;CHECK: s_wqm_b64 [[WQM:[^,]+]], [[CMP]]
;FIXME: This could just be: s_and_b64 exec, exec, [[WQM]]
;CHECK: v_cndmask_b32_e64 [[KILL:[^,]+]], -1.0, 1.0, [[WQM]]
;CHECK: v_cmpx_le_f32_e32 {{[^,]+}}, 0, [[KILL]]
;CHECK: s_endpgm
define amdgpu_ps void @kill(i32 %v0, i32 %v1) #1 {
main_body:
  %c = icmp eq i32 %v0, %v1
  %w = call i1 @llvm.amdgcn.wqm.vote(i1 %c)
  %r = select i1 %w, float 1.0, float -1.0
  call void @llvm.AMDGPU.kill(float %r)
  ret void
}

declare void @llvm.AMDGPU.kill(float) #1
declare i1 @llvm.amdgcn.wqm.vote(i1)

attributes #1 = { nounwind }
