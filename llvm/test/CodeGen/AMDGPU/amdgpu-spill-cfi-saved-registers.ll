; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=gfx900 -amdgpu-spill-sgpr-to-vgpr=true -amdgpu-spill-cfi-saved-regs=true -verify-machineinstrs < %s | FileCheck  -enable-var-scope -check-prefix=GCN -check-prefix=CI %s

; GCN-LABEL: {{^}}func_with_no_callee:
; GCN: ; %bb.0:
; GCN-NEXT: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT: v_writelane_b32 [[VGPR_REG:v[0-9]+]], exec_lo, 0
; GCN-NEXT: v_writelane_b32 [[VGPR_REG]], exec_hi, 1
; GCN-NEXT: v_writelane_b32 [[VGPR_REG]], s30, 2
; GCN-NEXT: v_writelane_b32 [[VGPR_REG]], s31, 3
; GCN-NOT:  v_readlane_b32 s31, [[VGPR_REG]], 3
; GCN-NOT:  v_readlane_b32 s30, [[VGPR_REG]], 2
; GCN-NOT:  v_readlane_b32 exec_hi, [[VGPR_REG]], 1
; GCN-NOT:  v_readlane_b32 exec_lo, [[VGPR_REG]], 0
define void @func_with_no_callee() #0 {
  ret void
}

declare void @external_void_func() #0

; GCN-LABEL: {{^}}with_func_call:
; GCN: ; %bb.0:
; GCN: buffer_store_dword [[VGPR_REG:v[0-9]+]], off, s[0:3], s32 offset:4
; GCN: v_writelane_b32 v32, s30, 0
; GCN-NEXT: s_load_dwordx2 s[4:5], s[4:5], 0x0
; GCN-NEXT: v_writelane_b32 [[VGPR_REG]], s31, 1
; GCN-NEXT: v_writelane_b32 [[VGPR_REG]], exec_lo, 2
; GCN-NEXT: v_writelane_b32 [[VGPR_REG]], exec_hi, 3
; GCN-NEXT: v_writelane_b32 [[VGPR_REG]], s30, 4
; GCN-NEXT: v_writelane_b32 [[VGPR_REG]], s31, 5
; GCN-NOT:  v_readlane_b32 s31, [[VGPR_REG]], 5
; GCN-NOT:  v_readlane_b32 s30, [[VGPR_REG]], 4
; GCN-NOT:  v_readlane_b32 exec_hi, [[VGPR_REG]], 3
; GCN-NOT:  v_readlane_b32 exec_lo, [[VGPR_REG]], 2
define void @with_func_call() #1 {
  call void @external_void_func()
  ret void
}

; GCN-LABEL: {{^}}func_sgpr_spill_no_calls:
; GCN: ; %bb.0:
; GCN-NEXT: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN: v_writelane_b32 [[VGPR_REG:v[0-9]+]], exec_lo, 16
; GCN-NEXT: v_writelane_b32 [[VGPR_REG]], exec_hi, 17
; GCN-NEXT: v_writelane_b32 [[VGPR_REG]], s30, 18
; GCN-NEXT: v_writelane_b32 [[VGPR_REG]], s31, 19
; GCN-NOT:  v_readlane_b32 s31, [[VGPR_REG]], 19
; GCN-NOT:  v_readlane_b32 s30, [[VGPR_REG]], 18
; GCN-NOT:  v_readlane_b32 exec_hi, [[VGPR_REG]], 17
; GCN-NOT:  v_readlane_b32 exec_lo, [[VGPR_REG]], 16
define void @func_sgpr_spill_no_calls(i32 %in) #0 {
  %wide.sgpr0 = call <16 x i32> asm sideeffect "; def $0", "=s" () #0
  %wide.sgpr1 = call <16 x i32> asm sideeffect "; def $0", "=s" () #0

  call void asm sideeffect "; use $0", "s"(<16 x i32> %wide.sgpr0) #0
  call void asm sideeffect "; use $0", "s"(<16 x i32> %wide.sgpr1) #0
  ret void
}

; GCN-LABEL: {{^}}func_mayclobber_v31:
; GCN: ; %bb.0:
; GCN-NEXT: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN: buffer_store_dword [[VGPR_REG:v[0-9]+]], off, s[0:3], s32 offset:4
; GCN:v_writelane_b32 [[VGPR_REG]], s30, 0
; GCN-NEXT: s_load_dwordx2 s[4:5], s[4:5], 0x0
; GCN-NEXT: v_writelane_b32 [[VGPR_REG]], s31, 1
; GCN-NEXT: v_writelane_b32 [[VGPR_REG]], exec_lo, 2
; GCN-NEXT: v_writelane_b32 [[VGPR_REG]], exec_hi, 3
; GCN-NEXT: v_writelane_b32 [[VGPR_REG]], s30, 4
; GCN-NEXT: v_writelane_b32 [[VGPR_REG]], s31, 5
; GCN-NOT:  v_readlane_b32 s31, [[VGPR_REG]], 5
; GCN-NOT:  v_readlane_b32 s30, [[VGPR_REG]], 4
; GCN-NOT:  v_readlane_b32 exec_hi, [[VGPR_REG]], 3
; GCN-NOT:  v_readlane_b32 exec_lo, [[VGPR_REG]], 2
define void @func_mayclobber_v31() #0 {
  %v31 = call i32 asm sideeffect "; def $0", "={v31}"()
  call void @external_void_func()
  call void asm sideeffect "; use $0", "{v31}"(i32 %v31)
  ret void
}

; GCN-LABEL: {{^}}test_kernel:
; GCN: ; %bb.0:
; GCN-NOT: v_writelane_b32 [[v[0-9]+]], exec_lo
; GCN-NOT: v_writelane_b32 [[v[0-9]+]], exec_hi
; GCN-NOT: v_writelane_b32 [[v[0-9]+]], s30
; GCN-NOT: v_writelane_b32 [[v[0-9]+]], s31
define amdgpu_kernel void @test_kernel() #0 {
  call void @external_void_func()
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind "no-frame-pointer-elim"="true" }
