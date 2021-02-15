; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=CHECK,WAVE64  %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=CHECK,WAVE32 %s

;CHECK-LABEL: {{^}}ret:
;CHECK: v_cmp_eq_u32_e32 [[CMP:[^,]+]], v0, v1
;WAVE64: s_wqm_b64 [[WQM:[^,]+]], [[CMP]]
;WAVE32: s_wqm_b32 [[WQM:[^,]+]], [[CMP]]
;CHECK: v_cndmask_b32_e64 v0, 0, 1.0, [[WQM]]
define amdgpu_ps float @ret(i32 %v0, i32 %v1) #1 {
main_body:
  %c = icmp eq i32 %v0, %v1
  %w = call i1 @llvm.amdgcn.wqm.vote(i1 %c)
  %r = select i1 %w, float 1.0, float 0.0
  ret float %r
}

;CHECK-LABEL: {{^}}true:
;WAVE64: s_wqm_b64
;WAVE32: s_wqm_b32
define amdgpu_ps float @true() #1 {
main_body:
  %w = call i1 @llvm.amdgcn.wqm.vote(i1 true)
  %r = select i1 %w, float 1.0, float 0.0
  ret float %r
}

;CHECK-LABEL: {{^}}false:
;WAVE64: s_wqm_b64
;WAVE32: s_wqm_b32
define amdgpu_ps float @false() #1 {
main_body:
  %w = call i1 @llvm.amdgcn.wqm.vote(i1 false)
  %r = select i1 %w, float 1.0, float 0.0
  ret float %r
}

;CHECK-LABEL: {{^}}kill:
;CHECK: v_cmp_eq_u32_e32 [[CMP:[^,]+]], v0, v1

;WAVE64: s_wqm_b64 [[WQM:[^,]+]], [[CMP]]
;WAVE64: s_and_b64 exec, exec, [[WQM]]

;WAVE32: s_wqm_b32 [[WQM:[^,]+]], [[CMP]]
;WAVE32: s_and_b32 exec_lo, exec_lo, [[WQM]]

;CHECK: s_endpgm
define amdgpu_ps void @kill(i32 %v0, i32 %v1) #1 {
main_body:
  %c = icmp eq i32 %v0, %v1
  %w = call i1 @llvm.amdgcn.wqm.vote(i1 %c)
  call void @llvm.amdgcn.kill(i1 %w)
  ret void
}

declare void @llvm.amdgcn.kill(i1) #1
declare i1 @llvm.amdgcn.wqm.vote(i1)

attributes #1 = { nounwind }
