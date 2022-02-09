; RUN: llc -mtriple=amdgcn-amd-amdhsa -verify-machineinstrs -enable-ipra -amdgpu-sroa=0 < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -verify-machineinstrs -amdgpu-sroa=0 < %s | FileCheck -check-prefix=GCN %s

; Kernels are not called, so there is no call preserved mask.
; GCN-LABEL: {{^}}kernel:
; GCN: flat_store_dword
define amdgpu_kernel void @kernel(i32 addrspace(1)* %out) #0 {
entry:
  store i32 0, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}func:
; GCN: ; NumVgprs: 8
define hidden void @func() #1 {
  call void asm sideeffect "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7}"() #0
  ret void
}

; GCN-LABEL: {{^}}kernel_call:
; GCN-NOT: buffer_store
; GCN-NOT: buffer_load
; GCN-NOT: readlane
; GCN-NOT: writelane
; GCN: flat_load_dword v8
; GCN: s_swappc_b64
; GCN-NOT: buffer_store
; GCN-NOT: buffer_load
; GCN-NOT: readlane
; GCN-NOT: writelane
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, v8

; GCN: ; NumSgprs: 37
; GCN: ; NumVgprs: 9
define amdgpu_kernel void @kernel_call() #0 {
  %vgpr = load volatile i32, i32 addrspace(1)* undef
  tail call void @func()
  store volatile i32 %vgpr, i32 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}func_regular_call:
; GCN-NOT: buffer_store
; GCN-NOT: buffer_load
; GCN-NOT: readlane
; GCN-NOT: writelane
; GCN: flat_load_dword v8
; GCN: s_swappc_b64
; GCN-NOT: buffer_store
; GCN-NOT: buffer_load
; GCN-NOT: readlane
; GCN-NOT: writelane
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, v8

; GCN: ; NumSgprs: 32
; GCN: ; NumVgprs: 9
define void @func_regular_call() #1 {
  %vgpr = load volatile i32, i32 addrspace(1)* undef
  tail call void @func()
  store volatile i32 %vgpr, i32 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}func_tail_call:
; GCN: s_waitcnt
; GCN-NEXT: s_getpc_b64 s[4:5]
; GCN-NEXT: s_add_u32 s4,
; GCN-NEXT: s_addc_u32 s5,
; GCN-NEXT: s_setpc_b64 s[4:5]

; GCN: ; NumSgprs: 32
; GCN: ; NumVgprs: 8
define void @func_tail_call() #1 {
  tail call void @func()
  ret void
}

; GCN-LABEL: {{^}}func_call_tail_call:
; GCN: flat_load_dword v8
; GCN: s_swappc_b64
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, v8
; GCN: s_setpc_b64

; GCN: ; NumSgprs: 32
; GCN: ; NumVgprs: 9
define void @func_call_tail_call() #1 {
  %vgpr = load volatile i32, i32 addrspace(1)* undef
  tail call void @func()
  store volatile i32 %vgpr, i32 addrspace(1)* undef
  tail call void @func()
  ret void
}

; GCN-LABEL: {{^}}void_func_void:
define void @void_func_void() noinline {
  ret void
}

; Make sure we don't get save/restore of FP between calls.
; GCN-LABEL: {{^}}test_funcx2:
; GCN-NOT: s5
; GCN-NOT: s32
define void @test_funcx2() #0 {
  call void @void_func_void()
  call void @void_func_void()
  ret void
}

; GCN-LABEL: {{^}}wombat:
define weak amdgpu_kernel void @wombat(i32* %arg, i32* %arg2) {
bb:
  call void @hoge() #0
  ret void
}

; Make sure we save/restore the return address around the call.
; Function Attrs: norecurse
define internal void @hoge() #2 {
bb:
; GCN-LABEL: {{^}}hoge:
; GCN-DAG: v_writelane_b32 [[CSR_VGPR:v[0-9]+]], s30,
; GCN-DAG: v_writelane_b32 [[CSR_VGPR]], s31,
; GCN: s_swappc_b64 s[30:31]
; GCN-DAG: v_readlane_b32 s4, [[CSR_VGPR]],
; GCN-DAG: v_readlane_b32 s5, [[CSR_VGPR]],
; GCN: s_waitcnt vmcnt(0)
; GCN: s_setpc_b64 s[4:5]
  call void @eggs()
  ret void
}

declare dso_local void @eggs()


attributes #0 = { nounwind }
attributes #1 = { nounwind noinline "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" }
attributes #2 = { norecurse }
