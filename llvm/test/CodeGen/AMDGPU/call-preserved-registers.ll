; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=fiji -enable-ipra=0 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=hawaii -enable-ipra=0 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -enable-ipra=0 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

declare hidden void @external_void_func_void() #0

; GCN-LABEL: {{^}}test_kernel_call_external_void_func_void_clobber_s30_s31_call_external_void_func_void:
; GCN: s_mov_b32 s33, s7
; GCN: s_mov_b32 s4, s33
; GCN-NEXT: s_getpc_b64 s[34:35]
; GCN-NEXT: s_add_u32 s34, s34,
; GCN-NEXT: s_addc_u32 s35, s35,
; GCN-NEXT: s_mov_b32 s32, s33
; GCN: s_swappc_b64 s[30:31], s[34:35]

; GCN-NEXT: s_mov_b32 s4, s33
; GCN-NEXT: #ASMSTART
; GCN-NEXT: #ASMEND
; GCN-NEXT: s_swappc_b64 s[30:31], s[34:35]
define amdgpu_kernel void @test_kernel_call_external_void_func_void_clobber_s30_s31_call_external_void_func_void() #0 {
  call void @external_void_func_void()
  call void asm sideeffect "", ""() #0
  call void @external_void_func_void()
  ret void
}

; GCN-LABEL: {{^}}test_func_call_external_void_func_void_clobber_s30_s31_call_external_void_func_void:
; GCN: v_writelane_b32 v32, s33, 0
; GCN: v_writelane_b32 v32, s34, 1
; GCN: v_writelane_b32 v32, s35, 2
; GCN: v_writelane_b32 v32, s36, 3
; GCN: v_writelane_b32 v32, s37, 4

; GCN: s_mov_b32 s33, s5
; GCN-NEXT: s_swappc_b64
; GCN-NEXT: s_mov_b32 s5, s33
; GCN-NEXT: s_mov_b32 s33, s5
; GCN-NEXT: ;;#ASMSTART
; GCN-NEXT: ;;#ASMEND
; GCN-NEXT: s_swappc_b64
; GCN-NEXT: s_mov_b32 s5, s33
; GCN: v_readlane_b32 s37, v32, 4
; GCN: v_readlane_b32 s36, v32, 3
; GCN: v_readlane_b32 s35, v32, 2
; GCN: v_readlane_b32 s34, v32, 1
; GCN: v_readlane_b32 s33, v32, 0
; GCN: s_setpc_b64
define void @test_func_call_external_void_func_void_clobber_s30_s31_call_external_void_func_void() #0 {
  call void @external_void_func_void()
  call void asm sideeffect "", ""() #0
  call void @external_void_func_void()
  ret void
}

; FIXME: Avoid extra restore of FP in between calls.
; GCN-LABEL: {{^}}test_func_call_external_void_funcx2:
; GCN: s_mov_b32 s33, s5
; GCN-NEXT: s_swappc_b64
; GCN-NEXT: s_mov_b32 s5, s33
; GCN-NEXT: s_mov_b32 s33, s5
; GCN-NEXT: s_swappc_b64
; GCN-NEXT: s_mov_b32 s5, s33
define void @test_func_call_external_void_funcx2() #0 {
  call void @external_void_func_void()
  call void @external_void_func_void()
  ret void
}

; GCN-LABEL: {{^}}void_func_void_clobber_s30_s31:
; GCN: s_waitcnt
; GCN-NEXT: s_mov_b64 [[SAVEPC:s\[[0-9]+:[0-9]+\]]], s[30:31]
; GCN-NEXT: #ASMSTART
; GCN: ; clobber
; GCN-NEXT: #ASMEND
; GCN-NEXT: s_mov_b64 s[30:31], [[SAVEPC]]
; GCN-NEXT: s_setpc_b64 s[30:31]
define void @void_func_void_clobber_s30_s31() #2 {
  call void asm sideeffect "; clobber", "~{s[30:31]}"() #0
  ret void
}

; GCN-LABEL: {{^}}void_func_void_clobber_vcc:
; GCN: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT: ;;#ASMSTART
; GCN-NEXT: ;;#ASMEND
; GCN-NEXT: s_setpc_b64 s[30:31]
define hidden void @void_func_void_clobber_vcc() #2 {
  call void asm sideeffect "", "~{VCC}"() #0
  ret void
}

; GCN-LABEL: {{^}}test_call_void_func_void_clobber_vcc:
; GCN: s_getpc_b64
; GCN-NEXT: s_add_u32
; GCN-NEXT: s_addc_u32
; GCN: s_mov_b64 s[34:35], vcc
; GCN-NEXT: s_swappc_b64
; GCN: s_mov_b64 vcc, s[34:35]
define amdgpu_kernel void @test_call_void_func_void_clobber_vcc(i32 addrspace(1)* %out) #0 {
  %vcc = call i64 asm sideeffect "; def $0", "={vcc}"()
  call void @void_func_void_clobber_vcc()
  %val0 = load volatile i32, i32 addrspace(1)* undef
  %val1 = load volatile i32, i32 addrspace(1)* undef
  call void asm sideeffect "; use $0", "{vcc}"(i64 %vcc)
  ret void
}

; GCN-LABEL: {{^}}test_call_void_func_void_mayclobber_s31:
; GCN: s_mov_b32 s33, s31
; GCN-NEXT: s_swappc_b64
; GCN-NEXT: s_mov_b32 s31, s33
define amdgpu_kernel void @test_call_void_func_void_mayclobber_s31(i32 addrspace(1)* %out) #0 {
  %s31 = call i32 asm sideeffect "; def $0", "={s31}"()
  call void @external_void_func_void()
  call void asm sideeffect "; use $0", "{s31}"(i32 %s31)
  ret void
}

; GCN-LABEL: {{^}}test_call_void_func_void_mayclobber_v31:
; GCN: v_mov_b32_e32 v32, v31
; GCN-NEXT: s_swappc_b64
; GCN-NEXT: v_mov_b32_e32 v31, v32
define amdgpu_kernel void @test_call_void_func_void_mayclobber_v31(i32 addrspace(1)* %out) #0 {
  %v31 = call i32 asm sideeffect "; def $0", "={v31}"()
  call void @external_void_func_void()
  call void asm sideeffect "; use $0", "{v31}"(i32 %v31)
  ret void
}

; GCN-LABEL: {{^}}test_call_void_func_void_preserves_s33:
; GCN: s_mov_b32 s34, s9
; GCN: s_mov_b32 s4, s34
; GCN-DAG: s_mov_b32 s32, s34
; GCN-DAG: ; def s33
; GCN-DAG: #ASMEND
; GCN-DAG: s_getpc_b64 s[6:7]
; GCN-DAG: s_add_u32 s6, s6, external_void_func_void@rel32@lo+4
; GCN-DAG: s_addc_u32 s7, s7, external_void_func_void@rel32@hi+4
; GCN-NEXT: s_swappc_b64 s[30:31], s[6:7]
; GCN-NEXT: ;;#ASMSTART
; GCN-NEXT: ; use s33
; GCN-NEXT: ;;#ASMEND
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @test_call_void_func_void_preserves_s33(i32 addrspace(1)* %out) #0 {
  %s33 = call i32 asm sideeffect "; def $0", "={s33}"()
  call void @external_void_func_void()
  call void asm sideeffect "; use $0", "{s33}"(i32 %s33)
  ret void
}

; GCN-LABEL: {{^}}test_call_void_func_void_preserves_v32:
; GCN: s_mov_b32 s33, s9
; GCN: s_mov_b32 s4, s33
; GCN-DAG: s_mov_b32 s32, s33
; GCN-DAG: ; def v32
; GCN-DAG: #ASMEND
; GCN-DAG: s_getpc_b64 s[6:7]
; GCN-DAG: s_add_u32 s6, s6, external_void_func_void@rel32@lo+4
; GCN-DAG: s_addc_u32 s7, s7, external_void_func_void@rel32@hi+4
; GCN-NEXT: s_swappc_b64 s[30:31], s[6:7]
; GCN-NEXT: ;;#ASMSTART
; GCN-NEXT: ; use v32
; GCN-NEXT: ;;#ASMEND
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @test_call_void_func_void_preserves_v32(i32 addrspace(1)* %out) #0 {
  %v32 = call i32 asm sideeffect "; def $0", "={v32}"()
  call void @external_void_func_void()
  call void asm sideeffect "; use $0", "{v32}"(i32 %v32)
  ret void
}

; GCN-LABEL: {{^}}void_func_void_clobber_s33:
; GCN: v_writelane_b32 v0, s33, 0
; GCN-NEXT: #ASMSTART
; GCN-NEXT: ; clobber
; GCN-NEXT: #ASMEND
; GCN-NEXT:	v_readlane_b32 s33, v0, 0
; GCN-NEXT: s_setpc_b64
define hidden void @void_func_void_clobber_s33() #2 {
  call void asm sideeffect "; clobber", "~{s33}"() #0
  ret void
}

; GCN-LABEL: {{^}}test_call_void_func_void_clobber_s33:
; GCN: s_mov_b32 s33, s7
; GCN: s_mov_b32 s4, s33
; GCN-NEXT: s_getpc_b64
; GCN-NEXT: s_add_u32
; GCN-NEXT: s_addc_u32
; GCN-NEXT: s_mov_b32 s32, s33
; GCN: s_swappc_b64
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @test_call_void_func_void_clobber_s33() #0 {
  call void @void_func_void_clobber_s33()
  ret void
}

; GCN-LABEL: {{^}}callee_saved_sgpr_func:
; GCN-NOT: s40
; GCN: v_writelane_b32 v32, s40
; GCN: s_swappc_b64
; GCN-NOT: s40
; GCN: ; use s40
; GCN-NOT: s40
; GCN: v_readlane_b32 s40, v32
; GCN-NOT: s40
define void @callee_saved_sgpr_func() #2 {
  %s40 = call i32 asm sideeffect "; def s40", "={s40}"() #0
  call void @external_void_func_void()
  call void asm sideeffect "; use $0", "s"(i32 %s40) #0
  ret void
}

; GCN-LABEL: {{^}}callee_saved_sgpr_kernel:
; GCN-NOT: s40
; GCN: ; def s40
; GCN-NOT: s40
; GCN: s_swappc_b64
; GCN-NOT: s40
; GCN: ; use s40
; GCN-NOT: s40
define amdgpu_kernel void @callee_saved_sgpr_kernel() #2 {
  %s40 = call i32 asm sideeffect "; def s40", "={s40}"() #0
  call void @external_void_func_void()
  call void asm sideeffect "; use $0", "s"(i32 %s40) #0
  ret void
}

; First call preserved VGPR is used so it can't be used for SGPR spills.
; GCN-LABEL: {{^}}callee_saved_sgpr_vgpr_func:
; GCN-NOT: s40
; GCN: v_writelane_b32 v33, s40
; GCN: s_swappc_b64
; GCN-NOT: s40
; GCN: ; use s40
; GCN-NOT: s40
; GCN: v_readlane_b32 s40, v33
; GCN-NOT: s40
define void @callee_saved_sgpr_vgpr_func() #2 {
  %s40 = call i32 asm sideeffect "; def s40", "={s40}"() #0
  %v32 = call i32 asm sideeffect "; def v32", "={v32}"() #0
  call void @external_void_func_void()
  call void asm sideeffect "; use $0", "s"(i32 %s40) #0
  call void asm sideeffect "; use $0", "v"(i32 %v32) #0
  ret void
}

; GCN-LABEL: {{^}}callee_saved_sgpr_vgpr_kernel:
; GCN-NOT: s40
; GCN: ; def s40
; GCN-NOT: s40
; GCN: s_swappc_b64
; GCN-NOT: s40
; GCN: ; use s40
; GCN-NOT: s40
define amdgpu_kernel void @callee_saved_sgpr_vgpr_kernel() #2 {
  %s40 = call i32 asm sideeffect "; def s40", "={s40}"() #0
  %v32 = call i32 asm sideeffect "; def v32", "={v32}"() #0
  call void @external_void_func_void()
  call void asm sideeffect "; use $0", "s"(i32 %s40) #0
  call void asm sideeffect "; use $0", "v"(i32 %v32) #0
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind noinline }
