; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=fiji -enable-ipra=0 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,MUBUF %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=hawaii -enable-ipra=0 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,MUBUF %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -enable-ipra=0 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,MUBUF %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -enable-ipra=0 -mattr=+enable-flat-scratch -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,FLATSCR %s

declare hidden void @external_void_func_void() #3

; GCN-LABEL: {{^}}test_kernel_call_external_void_func_void_clobber_s30_s31_call_external_void_func_void:
; GCN: s_getpc_b64 s[34:35]
; GCN-NEXT: s_add_u32 s34, s34,
; GCN-NEXT: s_addc_u32 s35, s35,
; GCN: s_swappc_b64 s[30:31], s[34:35]

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
; GCN: s_mov_b32 [[FP_SCRATCH_COPY:s[0-9]+]], s33
; MUBUF:   buffer_store_dword
; FLATSCR: scratch_store_dword
; GCN: v_writelane_b32 v40, [[FP_SCRATCH_COPY]], 4
; GCN: v_writelane_b32 v40, s30, 0
; GCN: v_writelane_b32 v40, s31, 1
; GCN: v_writelane_b32 v40, s34, 2
; GCN: v_writelane_b32 v40, s35, 3

; GCN: s_swappc_b64
; GCN-NEXT: ;;#ASMSTART
; GCN-NEXT: ;;#ASMEND
; GCN-NEXT: s_swappc_b64
; GCN: v_readlane_b32 s35, v40, 3
; GCN: v_readlane_b32 s34, v40, 2
; MUBUF-DAG:   v_readlane_b32 s31, v40, 1
; MUBUF-DAG:   v_readlane_b32 s30, v40, 0
; FLATSCR-DAG: v_readlane_b32 s31, v40, 1
; FLATSCR-DAG: v_readlane_b32 s30, v40, 0

; GCN: v_readlane_b32 [[FP_SCRATCH_COPY:s[0-9]+]], v40, 4
; MUBUF:   buffer_load_dword
; FLATSCR: scratch_load_dword
; GCN: s_mov_b32 s33, [[FP_SCRATCH_COPY]]
; GCN: s_setpc_b64 s[30:31]
define void @test_func_call_external_void_func_void_clobber_s30_s31_call_external_void_func_void() #0 {
  call void @external_void_func_void()
  call void asm sideeffect "", ""() #0
  call void @external_void_func_void()
  ret void
}

; GCN-LABEL: {{^}}test_func_call_external_void_funcx2:
; GCN: s_mov_b32 [[FP_SCRATCH_COPY:s[0-9]+]], s33
; GCN: s_mov_b32 s33, s32
; MUBUF:   buffer_store_dword v40
; FLATSCR: scratch_store_dword off, v40
; GCN: v_writelane_b32 v40, [[FP_SCRATCH_COPY]], 4
; MUBUF:   s_addk_i32 s32, 0x400
; FLATSCR: s_add_i32 s32, s32, 16

; GCN: s_swappc_b64
; GCN-NEXT: s_swappc_b64

; GCN: v_readlane_b32 [[FP_SCRATCH_COPY:s[0-9]+]], v40, 4
; MUBUF:   buffer_load_dword v40
; FLATSCR: scratch_load_dword v40
; GCN: s_mov_b32 s33, [[FP_SCRATCH_COPY]]
define void @test_func_call_external_void_funcx2() #0 {
  call void @external_void_func_void()
  call void @external_void_func_void()
  ret void
}

; GCN-LABEL: {{^}}void_func_void_clobber_s30_s31:
; GCN: s_waitcnt
; GCN: v_writelane_b32 v0, s30, 0
; GCN: v_writelane_b32 v0, s31, 1
; GCN-NEXT: #ASMSTART
; GCN: ; clobber
; GCN-NEXT: #ASMEND
; GCN: v_readlane_b32 s31, v0, 1
; GCN: v_readlane_b32 s30, v0, 0
; GCN: s_setpc_b64 s[30:31]
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
  call void asm sideeffect "", "~{vcc}"() #0
  ret void
}

; GCN-LABEL: {{^}}test_call_void_func_void_clobber_vcc:
; GCN: s_getpc_b64
; GCN-NEXT: s_add_u32
; GCN-NEXT: s_addc_u32
; GCN: s_mov_b64 s[34:35], vcc
; GCN-NEXT: s_swappc_b64
; GCN: s_mov_b64 vcc, s[34:35]
define amdgpu_kernel void @test_call_void_func_void_clobber_vcc(ptr addrspace(1) %out) #0 {
  %vcc = call i64 asm sideeffect "; def $0", "={vcc}"()
  call void @void_func_void_clobber_vcc()
  %val0 = load volatile i32, ptr addrspace(1) undef
  %val1 = load volatile i32, ptr addrspace(1) undef
  call void asm sideeffect "; use $0", "{vcc}"(i64 %vcc)
  ret void
}

; GCN-LABEL: {{^}}test_call_void_func_void_mayclobber_s31:
; GCN: s_mov_b32 s33, s31
; GCN: s_swappc_b64
; GCN-NEXT: s_mov_b32 s31, s33
define amdgpu_kernel void @test_call_void_func_void_mayclobber_s31(ptr addrspace(1) %out) #0 {
  %s31 = call i32 asm sideeffect "; def $0", "={s31}"()
  call void @external_void_func_void()
  call void asm sideeffect "; use $0", "{s31}"(i32 %s31)
  ret void
}

; GCN-LABEL: {{^}}test_call_void_func_void_mayclobber_v31:
; GCN: v_mov_b32_e32 v40, v31
; GCN: s_swappc_b64
; GCN-NEXT: v_mov_b32_e32 v31, v40
define amdgpu_kernel void @test_call_void_func_void_mayclobber_v31(ptr addrspace(1) %out) #0 {
  %v31 = call i32 asm sideeffect "; def $0", "={v31}"()
  call void @external_void_func_void()
  call void asm sideeffect "; use $0", "{v31}"(i32 %v31)
  ret void
}

; FIXME: What is the expected behavior for reserved registers here?

; GCN-LABEL: {{^}}test_call_void_func_void_preserves_s33:
; FLATSCR:      s_getpc_b64 s[0:1]
; FLATSCR-NEXT: s_add_u32 s0, s0, external_void_func_void@rel32@lo+4
; FLATSCR-NEXT: s_addc_u32 s1, s1, external_void_func_void@rel32@hi+12
; MUBUF:        s_getpc_b64 s[4:5]
; MUBUF-NEXT:   s_add_u32 s4, s4, external_void_func_void@rel32@lo+4
; MUBUF-NEXT:   s_addc_u32 s5, s5, external_void_func_void@rel32@hi+12

; GCN: #ASMSTART
; GCN-NEXT: ; def s33
; GCN-NEXT: #ASMEND

; GCN-NOT: s33

; FLATSCR: s_swappc_b64 s[30:31], s[0:1]
; MUBUF:   s_swappc_b64 s[30:31], s[4:5]

; GCN-NOT: s33

; GCN: ;;#ASMSTART
; GCN-NEXT: ; use s33
; GCN-NEXT: ;;#ASMEND
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @test_call_void_func_void_preserves_s33(ptr addrspace(1) %out) #0 {
  %s33 = call i32 asm sideeffect "; def $0", "={s33}"()
  call void @external_void_func_void()
  call void asm sideeffect "; use $0", "{s33}"(i32 %s33)
  ret void
}

; GCN-LABEL: {{^}}test_call_void_func_void_preserves_s34: {{.*}}
; GCN-NOT: s34

; FLATSCR:      s_getpc_b64 s[0:1]
; FLATSCR-NEXT: s_add_u32 s0, s0, external_void_func_void@rel32@lo+4
; FLATSCR-NEXT: s_addc_u32 s1, s1, external_void_func_void@rel32@hi+12
; MUBUF:        s_getpc_b64 s[4:5]
; MUBUF-NEXT:   s_add_u32 s4, s4, external_void_func_void@rel32@lo+4
; MUBUF-NEXT:   s_addc_u32 s5, s5, external_void_func_void@rel32@hi+12
; GCN: s_mov_b32 s32, 0

; GCN: ;;#ASMSTART
; GCN-NEXT: ; def s34
; GCN-NEXT: ;;#ASMEND

; GCN-NOT: s34

; MUBUF:   s_swappc_b64 s[30:31], s[4:5]
; FLATSCR: s_swappc_b64 s[30:31], s[0:1]

; GCN-NOT: s34

; GCN-NEXT: ;;#ASMSTART
; GCN-NEXT: ; use s34
; GCN-NEXT: ;;#ASMEND
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @test_call_void_func_void_preserves_s34(ptr addrspace(1) %out) #0 {
  %s34 = call i32 asm sideeffect "; def $0", "={s34}"()
  call void @external_void_func_void()
  call void asm sideeffect "; use $0", "{s34}"(i32 %s34)
  ret void
}

; GCN-LABEL: {{^}}test_call_void_func_void_preserves_v40: {{.*}}

; MUBUF: s_getpc_b64 s[4:5]
; MUBUF-NEXT:   s_add_u32 s4, s4, external_void_func_void@rel32@lo+4
; MUBUF-NEXT:   s_addc_u32 s5, s5, external_void_func_void@rel32@hi+12
; FLATSCR:      s_getpc_b64 s[0:1]
; FLATSCR-NEXT: s_add_u32 s0, s0, external_void_func_void@rel32@lo+4
; FLATSCR-NEXT: s_addc_u32 s1, s1, external_void_func_void@rel32@hi+12
; GCN: s_mov_b32 s32, 0

; GCN: ;;#ASMSTART
; GCN-NEXT: ; def v40
; GCN-NEXT: ;;#ASMEND

; GCN-NOT: v40

; MUBUF:   s_swappc_b64 s[30:31], s[4:5]
; FLATSCR: s_swappc_b64 s[30:31], s[0:1]

; GCN-NOT: v40

; GCN: ;;#ASMSTART
; GCN-NEXT: ; use v40
; GCN-NEXT: ;;#ASMEND
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @test_call_void_func_void_preserves_v40(ptr addrspace(1) %out) #0 {
  %v40 = call i32 asm sideeffect "; def $0", "={v40}"()
  call void @external_void_func_void()
  call void asm sideeffect "; use $0", "{v40}"(i32 %v40)
  ret void
}

; GCN-LABEL: {{^}}void_func_void_clobber_s33:
; GCN: v_writelane_b32 v0, s33, 0
; GCN-NEXT: #ASMSTART
; GCN-NEXT: ; clobber
; GCN-NEXT: #ASMEND
; GCN-NEXT:	v_readlane_b32 s33, v0, 0
; GCN: s_setpc_b64
define hidden void @void_func_void_clobber_s33() #2 {
  call void asm sideeffect "; clobber", "~{s33}"() #0
  ret void
}

; GCN-LABEL: {{^}}void_func_void_clobber_s34:
; GCN: v_writelane_b32 v0, s34, 0
; GCN-NEXT: #ASMSTART
; GCN-NEXT: ; clobber
; GCN-NEXT: #ASMEND
; GCN-NEXT:	v_readlane_b32 s34, v0, 0
; GCN: s_setpc_b64
define hidden void @void_func_void_clobber_s34() #2 {
  call void asm sideeffect "; clobber", "~{s34}"() #0
  ret void
}

; GCN-LABEL: {{^}}test_call_void_func_void_clobber_s33:
; GCN: s_getpc_b64
; GCN-NEXT: s_add_u32
; GCN-NEXT: s_addc_u32
; GCN: s_mov_b32 s32, 0
; GCN: s_swappc_b64
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @test_call_void_func_void_clobber_s33() #0 {
  call void @void_func_void_clobber_s33()
  ret void
}

; GCN-LABEL: {{^}}test_call_void_func_void_clobber_s34:
; GCN: s_getpc_b64
; GCN-NEXT: s_add_u32
; GCN-NEXT: s_addc_u32
; GCN: s_mov_b32 s32, 0
; GCN: s_swappc_b64
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @test_call_void_func_void_clobber_s34() #0 {
  call void @void_func_void_clobber_s34()
  ret void
}

; GCN-LABEL: {{^}}callee_saved_sgpr_func:
; GCN-NOT: s40
; GCN: v_writelane_b32 v40, s40
; GCN: s_swappc_b64
; GCN-NOT: s40
; GCN: ; use s40
; GCN-NOT: s40
; GCN: v_readlane_b32 s40, v40
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
; GCN: v_writelane_b32 v41, s40
; GCN: s_swappc_b64
; GCN-NOT: s40
; GCN: ; use s40
; GCN-NOT: s40
; GCN: v_readlane_b32 s40, v41
; GCN-NOT: s40
define void @callee_saved_sgpr_vgpr_func() #2 {
  %s40 = call i32 asm sideeffect "; def s40", "={s40}"() #0
  %v40 = call i32 asm sideeffect "; def v40", "={v40}"() #0
  call void @external_void_func_void()
  call void asm sideeffect "; use $0", "s"(i32 %s40) #0
  call void asm sideeffect "; use $0", "v"(i32 %v40) #0
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
attributes #3 = { nounwind "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" }
