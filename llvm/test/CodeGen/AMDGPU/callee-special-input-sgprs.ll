; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=kaveri -mattr=-code-object-v3 -enable-ipra=0 -verify-machineinstrs < %s | FileCheck -allow-deprecated-dag-overlap -enable-var-scope -check-prefixes=GCN,CIVI %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -mattr=-code-object-v3 -enable-ipra=0 -verify-machineinstrs < %s | FileCheck -allow-deprecated-dag-overlap -enable-var-scope -check-prefixes=GCN,GFX9 %s

; GCN-LABEL: {{^}}use_dispatch_ptr:
; GCN: v_mov_b32_e32 v[[LO:[0-9]+]], s4
; GCN: v_mov_b32_e32 v[[HI:[0-9]+]], s5
; GCN: {{flat|global}}_load_dword v{{[0-9]+}}, v{{\[}}[[LO]]:[[HI]]{{\]}}
define hidden void @use_dispatch_ptr() #1 {
  %dispatch_ptr = call noalias i8 addrspace(4)* @llvm.amdgcn.dispatch.ptr() #0
  %header_ptr = bitcast i8 addrspace(4)* %dispatch_ptr to i32 addrspace(4)*
  %value = load volatile i32, i32 addrspace(4)* %header_ptr
  ret void
}

; GCN-LABEL: {{^}}kern_indirect_use_dispatch_ptr:
; GCN: enable_sgpr_dispatch_ptr = 1
; GCN-NOT: s[4:5]
; GCN-NOT: s4
; GCN-NOT: s5
define amdgpu_kernel void @kern_indirect_use_dispatch_ptr(i32) #1 {
  call void @use_dispatch_ptr()
  ret void
}

; GCN-LABEL: {{^}}use_queue_ptr:
; GCN: v_mov_b32_e32 v[[LO:[0-9]+]], s4
; GCN: v_mov_b32_e32 v[[HI:[0-9]+]], s5
; GCN: {{flat|global}}_load_dword v{{[0-9]+}}, v{{\[}}[[LO]]:[[HI]]{{\]}}
define hidden void @use_queue_ptr() #1 {
  %queue_ptr = call noalias i8 addrspace(4)* @llvm.amdgcn.queue.ptr() #0
  %header_ptr = bitcast i8 addrspace(4)* %queue_ptr to i32 addrspace(4)*
  %value = load volatile i32, i32 addrspace(4)* %header_ptr
  ret void
}

; GCN-LABEL: {{^}}kern_indirect_use_queue_ptr:
; GCN: enable_sgpr_queue_ptr = 1
; GCN-NOT: s[4:5]
; GCN-NOT: s4
; GCN-NOT: s5
define amdgpu_kernel void @kern_indirect_use_queue_ptr(i32) #1 {
  call void @use_queue_ptr()
  ret void
}

; GCN-LABEL: {{^}}use_queue_ptr_addrspacecast:
; CIVI: flat_load_dword v[[HI:[0-9]+]], v[0:1]
; GFX9: s_getreg_b32 [[APERTURE_LOAD:s[0-9]+]]
; CIVI: v_mov_b32_e32 v[[LO:[0-9]+]], 16
; GFX9: v_mov_b32_e32 v[[HI:[0-9]+]], [[APERTURE_LOAD]]
; GFX9: {{flat|global}}_store_dword v{{\[[0-9]+}}:[[HI]]{{\]}}
; CIVI: {{flat|global}}_store_dword v{{\[}}[[LO]]:[[HI]]{{\]}}
define hidden void @use_queue_ptr_addrspacecast() #1 {
  %asc = addrspacecast i32 addrspace(3)* inttoptr (i32 16 to i32 addrspace(3)*) to i32*
  store volatile i32 0, i32* %asc
  ret void
}

; GCN-LABEL: {{^}}kern_indirect_use_queue_ptr_addrspacecast:
; CIVI: enable_sgpr_queue_ptr = 1
; CIVI-NOT: s[4:5]
; CIVI-NOT: s4
; CIVI-NOT: s5
define amdgpu_kernel void @kern_indirect_use_queue_ptr_addrspacecast(i32) #1 {
  call void @use_queue_ptr_addrspacecast()
  ret void
}

; Not really supported in callable functions.
; GCN-LABEL: {{^}}use_kernarg_segment_ptr:
; GCN: s_mov_b64 [[PTR:s\[[0-9]+:[0-9]+\]]], 0{{$}}
; GCN: s_load_dword s{{[0-9]+}}, [[PTR]], 0x0{{$}}
define hidden void @use_kernarg_segment_ptr() #1 {
  %kernarg_segment_ptr = call noalias i8 addrspace(4)* @llvm.amdgcn.kernarg.segment.ptr() #0
  %header_ptr = bitcast i8 addrspace(4)* %kernarg_segment_ptr to i32 addrspace(4)*
  %value = load volatile i32, i32 addrspace(4)* %header_ptr
  ret void
}

; GCN-LABEL: {{^}}kern_indirect_use_kernarg_segment_ptr:
; GCN: enable_sgpr_kernarg_segment_ptr = 1
define amdgpu_kernel void @kern_indirect_use_kernarg_segment_ptr(i32) #1 {
  call void @use_kernarg_segment_ptr()
  ret void
}

; GCN-LABEL: {{^}}use_dispatch_id:
; GCN: ; use s[4:5]
define hidden void @use_dispatch_id() #1 {
  %id = call i64 @llvm.amdgcn.dispatch.id()
  call void asm sideeffect "; use $0", "s"(i64 %id)
  ret void
}

; No kernarg segment so that there is a mov to check. With kernarg
; pointer enabled, it happens to end up in the right place anyway.

; GCN-LABEL: {{^}}kern_indirect_use_dispatch_id:
; GCN: enable_sgpr_dispatch_id = 1
; GCN-NOT: s[4:5]
; GCN-NOT: s4
; GCN-NOT: s5
define amdgpu_kernel void @kern_indirect_use_dispatch_id() #1 {
  call void @use_dispatch_id()
  ret void
}

; GCN-LABEL: {{^}}use_workgroup_id_x:
; GCN: s_waitcnt
; GCN: ; use s4
define hidden void @use_workgroup_id_x() #1 {
  %val = call i32 @llvm.amdgcn.workgroup.id.x()
  call void asm sideeffect "; use $0", "s"(i32 %val)
  ret void
}

; GCN-LABEL: {{^}}use_stack_workgroup_id_x:
; GCN: s_waitcnt
; GCN-NOT: s32
; GCN: buffer_store_dword v0, off, s[0:3], s32{{$}}
; GCN: ; use s4
; GCN: s_setpc_b64
define hidden void @use_stack_workgroup_id_x() #1 {
  %alloca = alloca i32, addrspace(5)
  store volatile i32 0, i32 addrspace(5)* %alloca
  %val = call i32 @llvm.amdgcn.workgroup.id.x()
  call void asm sideeffect "; use $0", "s"(i32 %val)
  ret void
}

; GCN-LABEL: {{^}}use_workgroup_id_y:
; GCN: s_waitcnt
; GCN: ; use s4
define hidden void @use_workgroup_id_y() #1 {
  %val = call i32 @llvm.amdgcn.workgroup.id.y()
  call void asm sideeffect "; use $0", "s"(i32 %val)
  ret void
}

; GCN-LABEL: {{^}}use_workgroup_id_z:
; GCN: s_waitcnt
; GCN: ; use s4
define hidden void @use_workgroup_id_z() #1 {
  %val = call i32 @llvm.amdgcn.workgroup.id.z()
  call void asm sideeffect "; use $0", "s"(i32 %val)
  ret void
}

; GCN-LABEL: {{^}}use_workgroup_id_xy:
; GCN: ; use s4
; GCN: ; use s5
define hidden void @use_workgroup_id_xy() #1 {
  %val0 = call i32 @llvm.amdgcn.workgroup.id.x()
  %val1 = call i32 @llvm.amdgcn.workgroup.id.y()
  call void asm sideeffect "; use $0", "s"(i32 %val0)
  call void asm sideeffect "; use $0", "s"(i32 %val1)
  ret void
}

; GCN-LABEL: {{^}}use_workgroup_id_xyz:
; GCN: ; use s4
; GCN: ; use s5
; GCN: ; use s6
define hidden void @use_workgroup_id_xyz() #1 {
  %val0 = call i32 @llvm.amdgcn.workgroup.id.x()
  %val1 = call i32 @llvm.amdgcn.workgroup.id.y()
  %val2 = call i32 @llvm.amdgcn.workgroup.id.z()
  call void asm sideeffect "; use $0", "s"(i32 %val0)
  call void asm sideeffect "; use $0", "s"(i32 %val1)
  call void asm sideeffect "; use $0", "s"(i32 %val2)
  ret void
}

; GCN-LABEL: {{^}}use_workgroup_id_xz:
; GCN: ; use s4
; GCN: ; use s5
define hidden void @use_workgroup_id_xz() #1 {
  %val0 = call i32 @llvm.amdgcn.workgroup.id.x()
  %val1 = call i32 @llvm.amdgcn.workgroup.id.z()
  call void asm sideeffect "; use $0", "s"(i32 %val0)
  call void asm sideeffect "; use $0", "s"(i32 %val1)
  ret void
}

; GCN-LABEL: {{^}}use_workgroup_id_yz:
; GCN: ; use s4
; GCN: ; use s5
define hidden void @use_workgroup_id_yz() #1 {
  %val0 = call i32 @llvm.amdgcn.workgroup.id.y()
  %val1 = call i32 @llvm.amdgcn.workgroup.id.z()
  call void asm sideeffect "; use $0", "s"(i32 %val0)
  call void asm sideeffect "; use $0", "s"(i32 %val1)
  ret void
}

; GCN-LABEL: {{^}}kern_indirect_use_workgroup_id_x:
; GCN: enable_sgpr_workgroup_id_x = 1
; GCN: enable_sgpr_workgroup_id_y = 0
; GCN: enable_sgpr_workgroup_id_z = 0

; GCN-NOT: s6
; GCN: s_mov_b32 s4, s6
; GCN-NEXT: s_getpc_b64 s[6:7]
; GCN-NEXT: s_add_u32 s6, s6, use_workgroup_id_x@rel32@lo+4
; GCN-NEXT: s_addc_u32 s7, s7, use_workgroup_id_x@rel32@hi+4
; GCN: s_mov_b32 s32, 0
; GCN: s_swappc_b64
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @kern_indirect_use_workgroup_id_x() #1 {
  call void @use_workgroup_id_x()
  ret void
}

; GCN-LABEL: {{^}}kern_indirect_use_workgroup_id_y:
; GCN: enable_sgpr_workgroup_id_x = 1
; GCN: enable_sgpr_workgroup_id_y = 1
; GCN: enable_sgpr_workgroup_id_z = 0

; GCN: s_mov_b32 s4, s7
; GCN: s_mov_b32 s32, 0
; GCN: s_swappc_b64
define amdgpu_kernel void @kern_indirect_use_workgroup_id_y() #1 {
  call void @use_workgroup_id_y()
  ret void
}

; GCN-LABEL: {{^}}kern_indirect_use_workgroup_id_z:
; GCN: enable_sgpr_workgroup_id_x = 1
; GCN: enable_sgpr_workgroup_id_y = 0
; GCN: enable_sgpr_workgroup_id_z = 1

; GCN: s_mov_b32 s4, s7

; GCN: s_mov_b32 s32, 0
; GCN: s_swappc_b64
define amdgpu_kernel void @kern_indirect_use_workgroup_id_z() #1 {
  call void @use_workgroup_id_z()
  ret void
}

; GCN-LABEL: {{^}}kern_indirect_use_workgroup_id_xy:
; GCN: enable_sgpr_workgroup_id_x = 1
; GCN: enable_sgpr_workgroup_id_y = 1
; GCN: enable_sgpr_workgroup_id_z = 0

; GCN: s_mov_b32 s5, s7
; GCN: s_mov_b32 s4, s6

; GCN: s_mov_b32 s32, 0
; GCN: s_swappc_b64
define amdgpu_kernel void @kern_indirect_use_workgroup_id_xy() #1 {
  call void @use_workgroup_id_xy()
  ret void
}

; GCN-LABEL: {{^}}kern_indirect_use_workgroup_id_xyz:
; GCN: enable_sgpr_workgroup_id_x = 1
; GCN: enable_sgpr_workgroup_id_y = 1
; GCN: enable_sgpr_workgroup_id_z = 1

; GCN: s_mov_b32 s4, s6
; GCN: s_mov_b32 s5, s7
; GCN: s_mov_b32 s6, s8

; GCN: s_mov_b32 s32, 0
; GCN: s_swappc_b64
define amdgpu_kernel void @kern_indirect_use_workgroup_id_xyz() #1 {
  call void @use_workgroup_id_xyz()
  ret void
}

; GCN-LABEL: {{^}}kern_indirect_use_workgroup_id_xz:
; GCN: enable_sgpr_workgroup_id_x = 1
; GCN: enable_sgpr_workgroup_id_y = 0
; GCN: enable_sgpr_workgroup_id_z = 1

; GCN: s_mov_b32 s5, s7
; GCN: s_mov_b32 s4, s6

; GCN: s_mov_b32 s32, 0
; GCN: s_swappc_b64
define amdgpu_kernel void @kern_indirect_use_workgroup_id_xz() #1 {
  call void @use_workgroup_id_xz()
  ret void
}

; GCN-LABEL: {{^}}kern_indirect_use_workgroup_id_yz:
; GCN: enable_sgpr_workgroup_id_x = 1
; GCN: enable_sgpr_workgroup_id_y = 1
; GCN: enable_sgpr_workgroup_id_z = 1

; GCN: s_mov_b32 s4, s7
; GCN: s_mov_b32 s5, s8

; GCN: s_mov_b32 s32, 0
; GCN: s_swappc_b64
define amdgpu_kernel void @kern_indirect_use_workgroup_id_yz() #1 {
  call void @use_workgroup_id_yz()
  ret void
}

; Argument is in right place already
; GCN-LABEL: {{^}}func_indirect_use_workgroup_id_x:
; GCN-NOT: s4
; GCN: v_readlane_b32 s4, v32, 0
define hidden void @func_indirect_use_workgroup_id_x() #1 {
  call void @use_workgroup_id_x()
  ret void
}

; GCN-LABEL: {{^}}func_indirect_use_workgroup_id_y:
; GCN-NOT: s4
; GCN: v_readlane_b32 s4, v32, 0
define hidden void @func_indirect_use_workgroup_id_y() #1 {
  call void @use_workgroup_id_y()
  ret void
}

; GCN-LABEL: {{^}}func_indirect_use_workgroup_id_z:
; GCN-NOT: s4
; GCN: v_readlane_b32 s4, v32, 0
define hidden void @func_indirect_use_workgroup_id_z() #1 {
  call void @use_workgroup_id_z()
  ret void
}

; GCN-LABEL: {{^}}other_arg_use_workgroup_id_x:
; GCN: {{flat|global}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, v0
; GCN: ; use s4
define hidden void @other_arg_use_workgroup_id_x(i32 %arg0) #1 {
  %val = call i32 @llvm.amdgcn.workgroup.id.x()
  store volatile i32 %arg0, i32 addrspace(1)* undef
  call void asm sideeffect "; use $0", "s"(i32 %val)
  ret void
}

; GCN-LABEL: {{^}}other_arg_use_workgroup_id_y:
; GCN: {{flat|global}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, v0
; GCN: ; use s4
define hidden void @other_arg_use_workgroup_id_y(i32 %arg0) #1 {
  %val = call i32 @llvm.amdgcn.workgroup.id.y()
  store volatile i32 %arg0, i32 addrspace(1)* undef
  call void asm sideeffect "; use $0", "s"(i32 %val)
  ret void
}

; GCN-LABEL: {{^}}other_arg_use_workgroup_id_z:
; GCN: {{flat|global}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, v0
; GCN: ; use s4
define hidden void @other_arg_use_workgroup_id_z(i32 %arg0) #1 {
  %val = call i32 @llvm.amdgcn.workgroup.id.z()
  store volatile i32 %arg0, i32 addrspace(1)* undef
  call void asm sideeffect "; use $0", "s"(i32 %val)
  ret void
}

; GCN-LABEL: {{^}}kern_indirect_other_arg_use_workgroup_id_x:
; GCN: enable_sgpr_workgroup_id_x = 1
; GCN: enable_sgpr_workgroup_id_y = 0
; GCN: enable_sgpr_workgroup_id_z = 0

; GCN-DAG: v_mov_b32_e32 v0, 0x22b
; GCN-DAG: s_mov_b32 s4, s6

; GCN-DAG: s_mov_b32 s32, 0
; GCN-NOT: s4
; GCN: s_swappc_b64
define amdgpu_kernel void @kern_indirect_other_arg_use_workgroup_id_x() #1 {
  call void @other_arg_use_workgroup_id_x(i32 555)
  ret void
}

; GCN-LABEL: {{^}}kern_indirect_other_arg_use_workgroup_id_y:
; GCN: enable_sgpr_workgroup_id_x = 1
; GCN: enable_sgpr_workgroup_id_y = 1
; GCN: enable_sgpr_workgroup_id_z = 0

; GCN-DAG: v_mov_b32_e32 v0, 0x22b
; GCN-DAG: s_mov_b32 s4, s7

; GCN-DAG: s_mov_b32 s32, 0
; GCN: s_swappc_b64
define amdgpu_kernel void @kern_indirect_other_arg_use_workgroup_id_y() #1 {
  call void @other_arg_use_workgroup_id_y(i32 555)
  ret void
}

; GCN-LABEL: {{^}}kern_indirect_other_arg_use_workgroup_id_z:
; GCN: enable_sgpr_workgroup_id_x = 1
; GCN: enable_sgpr_workgroup_id_y = 0
; GCN: enable_sgpr_workgroup_id_z = 1

; GCN-DAG: v_mov_b32_e32 v0, 0x22b

; GCN: s_mov_b32 s32, 0
; GCN: s_swappc_b64
define amdgpu_kernel void @kern_indirect_other_arg_use_workgroup_id_z() #1 {
  call void @other_arg_use_workgroup_id_z(i32 555)
  ret void
}

; GCN-LABEL: {{^}}use_every_sgpr_input:
; GCN: buffer_store_dword v{{[0-9]+}}, off, s[0:3], s32{{$}}
; GCN: v_mov_b32_e32 v[[LO:[0-9]+]], s4
; GCN: v_mov_b32_e32 v[[HI:[0-9]+]], s5
; GCN: {{flat|global}}_load_dword v{{[0-9]+}}, v{{\[}}[[LO]]:[[HI]]{{\]}}
; GCN: v_mov_b32_e32 v[[LO:[0-9]+]], s6
; GCN: v_mov_b32_e32 v[[HI:[0-9]+]], s7
; GCN: {{flat|global}}_load_dword v{{[0-9]+}}, v{{\[}}[[LO]]:[[HI]]{{\]}}
; GCN: v_mov_b32_e32 v[[LO:[0-9]+]], s8
; GCN: v_mov_b32_e32 v[[HI:[0-9]+]], s9
; GCN: {{flat|global}}_load_dword v{{[0-9]+}}, v{{\[}}[[LO]]:[[HI]]{{\]}}
; GCN: ; use s[10:11]
; GCN: ; use s12
; GCN: ; use s13
; GCN: ; use s14
define hidden void @use_every_sgpr_input() #1 {
  %alloca = alloca i32, align 4, addrspace(5)
  store volatile i32 0, i32 addrspace(5)* %alloca

  %dispatch_ptr = call noalias i8 addrspace(4)* @llvm.amdgcn.dispatch.ptr() #0
  %dispatch_ptr.bc = bitcast i8 addrspace(4)* %dispatch_ptr to i32 addrspace(4)*
  %val0 = load volatile i32, i32 addrspace(4)* %dispatch_ptr.bc

  %queue_ptr = call noalias i8 addrspace(4)* @llvm.amdgcn.queue.ptr() #0
  %queue_ptr.bc = bitcast i8 addrspace(4)* %queue_ptr to i32 addrspace(4)*
  %val1 = load volatile i32, i32 addrspace(4)* %queue_ptr.bc

  %implicitarg.ptr = call noalias i8 addrspace(4)* @llvm.amdgcn.implicitarg.ptr() #0
  %implicitarg.ptr.bc = bitcast i8 addrspace(4)* %implicitarg.ptr to i32 addrspace(4)*
  %val2 = load volatile i32, i32 addrspace(4)* %implicitarg.ptr.bc

  %val3 = call i64 @llvm.amdgcn.dispatch.id()
  call void asm sideeffect "; use $0", "s"(i64 %val3)

  %val4 = call i32 @llvm.amdgcn.workgroup.id.x()
  call void asm sideeffect "; use $0", "s"(i32 %val4)

  %val5 = call i32 @llvm.amdgcn.workgroup.id.y()
  call void asm sideeffect "; use $0", "s"(i32 %val5)

  %val6 = call i32 @llvm.amdgcn.workgroup.id.z()
  call void asm sideeffect "; use $0", "s"(i32 %val6)

  ret void
}

; GCN-LABEL: {{^}}kern_indirect_use_every_sgpr_input:
; GCN: enable_sgpr_workgroup_id_x = 1
; GCN: enable_sgpr_workgroup_id_y = 1
; GCN: enable_sgpr_workgroup_id_z = 1
; GCN: enable_sgpr_workgroup_info = 0

; GCN: enable_sgpr_private_segment_buffer = 1
; GCN: enable_sgpr_dispatch_ptr = 1
; GCN: enable_sgpr_queue_ptr = 1
; GCN: enable_sgpr_kernarg_segment_ptr = 1
; GCN: enable_sgpr_dispatch_id = 1
; GCN: enable_sgpr_flat_scratch_init = 1

; GCN: s_mov_b32 s12, s14
; GCN: s_mov_b32 s13, s15
; GCN: s_mov_b32 s14, s16
; GCN: s_mov_b32 s32, 0
; GCN: s_swappc_b64
define amdgpu_kernel void @kern_indirect_use_every_sgpr_input() #1 {
  call void @use_every_sgpr_input()
  ret void
}

; GCN-LABEL: {{^}}func_indirect_use_every_sgpr_input:
; GCN-NOT: s6
; GCN-NOT: s7
; GCN-NOT: s8
; GCN-NOT: s9
; GCN-NOT: s10
; GCN-NOT: s11
; GCN-NOT: s12
; GCN-NOT: s13
; GCN-NOT: s[6:7]
; GCN-NOT: s[8:9]
; GCN-NOT: s[10:11]
; GCN-NOT: s[12:13]
; GCN: s_or_saveexec_b64 s[16:17], -1
define hidden void @func_indirect_use_every_sgpr_input() #1 {
  call void @use_every_sgpr_input()
  ret void
}

; GCN-LABEL: {{^}}func_use_every_sgpr_input_call_use_workgroup_id_xyz:
; GCN: s_mov_b32 s4, s12
; GCN: s_mov_b32 s5, s13
; GCN: s_mov_b32 s6, s14
; GCN: ; use s[10:11]
; GCN: ; use s12
; GCN: ; use s13
; GCN: ; use s14

; GCN: s_swappc_b64
define hidden void @func_use_every_sgpr_input_call_use_workgroup_id_xyz() #1 {
  %alloca = alloca i32, align 4, addrspace(5)
  store volatile i32 0, i32 addrspace(5)* %alloca

  %dispatch_ptr = call noalias i8 addrspace(4)* @llvm.amdgcn.dispatch.ptr() #0
  %dispatch_ptr.bc = bitcast i8 addrspace(4)* %dispatch_ptr to i32 addrspace(4)*
  %val0 = load volatile i32, i32 addrspace(4)* %dispatch_ptr.bc

  %queue_ptr = call noalias i8 addrspace(4)* @llvm.amdgcn.queue.ptr() #0
  %queue_ptr.bc = bitcast i8 addrspace(4)* %queue_ptr to i32 addrspace(4)*
  %val1 = load volatile i32, i32 addrspace(4)* %queue_ptr.bc

  %implicitarg.ptr = call noalias i8 addrspace(4)* @llvm.amdgcn.implicitarg.ptr() #0
  %implicitarg.ptr.bc = bitcast i8 addrspace(4)* %implicitarg.ptr to i32 addrspace(4)*
  %val2 = load volatile i32, i32 addrspace(4)* %implicitarg.ptr.bc

  %val3 = call i64 @llvm.amdgcn.dispatch.id()
  call void asm sideeffect "; use $0", "s"(i64 %val3)

  %val4 = call i32 @llvm.amdgcn.workgroup.id.x()
  call void asm sideeffect "; use $0", "s"(i32 %val4)

  %val5 = call i32 @llvm.amdgcn.workgroup.id.y()
  call void asm sideeffect "; use $0", "s"(i32 %val5)

  %val6 = call i32 @llvm.amdgcn.workgroup.id.z()
  call void asm sideeffect "; use $0", "s"(i32 %val6)

  call void @use_workgroup_id_xyz()
  ret void
}

; GCN-LABEL: {{^}}func_use_every_sgpr_input_call_use_workgroup_id_xyz_spill:
; GCN-DAG: s_mov_b32 s33, s32
; GCN-DAG: s_add_u32 s32, s32, 0x400
; GCN-DAG: s_mov_b64 s{{\[}}[[LO_X:[0-9]+]]{{\:}}[[HI_X:[0-9]+]]{{\]}}, s[4:5]
; GCN-DAG: s_mov_b64 s{{\[}}[[LO_Y:[0-9]+]]{{\:}}[[HI_Y:[0-9]+]]{{\]}}, s[6:7]


; GCN: s_mov_b32 s4, s12
; GCN: s_mov_b32 s5, s13
; GCN: s_mov_b32 s6, s14

; GCN: s_mov_b64 s{{\[}}[[LO_Z:[0-9]+]]{{\:}}[[HI_Z:[0-9]+]]{{\]}}, s[8:9]

; GCN-DAG: s_mov_b32 [[SAVE_X:s[0-57-9][0-9]*]], s12
; GCN-DAG: s_mov_b32 [[SAVE_Y:s[0-57-9][0-9]*]], s13
; GCN-DAG: s_mov_b32 [[SAVE_Z:s[0-68-9][0-9]*]], s14



; GCN: s_swappc_b64

; GCN-DAG: buffer_store_dword v{{[0-9]+}}, off, s[0:3], s33{{$}}
; GCN-DAG: v_mov_b32_e32 v[[LO1:[0-9]+]], s[[LO_X]]
; GCN-DAG: v_mov_b32_e32 v[[HI1:[0-9]+]], s[[HI_X]]
; GCN-DAG: {{flat|global}}_load_dword v{{[0-9]+}}, v{{\[}}[[LO1]]:[[HI1]]{{\]}}
; GCN-DAG: v_mov_b32_e32 v[[LO2:[0-9]+]], s[[LO_Y]]
; GCN-DAG: v_mov_b32_e32 v[[HI2:[0-9]+]], s[[HI_Y]]
; GCN-DAG: {{flat|global}}_load_dword v{{[0-9]+}}, v{{\[}}[[LO2]]:[[HI2]]{{\]}}
; GCN-DAG: v_mov_b32_e32 v[[LO3:[0-9]+]], s[[LO_Z]]
; GCN-DAG: v_mov_b32_e32 v[[HI3:[0-9]+]], s[[HI_Z]]
; GCN-DAG: {{flat|global}}_load_dword v{{[0-9]+}}, v{{\[}}[[LO3]]:[[HI3]]{{\]}}
; GCN: ; use
; GCN: ; use [[SAVE_X]]
; GCN: ; use [[SAVE_Y]]
; GCN: ; use [[SAVE_Z]]
define hidden void @func_use_every_sgpr_input_call_use_workgroup_id_xyz_spill() #1 {
  %alloca = alloca i32, align 4, addrspace(5)
  call void @use_workgroup_id_xyz()

  store volatile i32 0, i32 addrspace(5)* %alloca

  %dispatch_ptr = call noalias i8 addrspace(4)* @llvm.amdgcn.dispatch.ptr() #0
  %dispatch_ptr.bc = bitcast i8 addrspace(4)* %dispatch_ptr to i32 addrspace(4)*
  %val0 = load volatile i32, i32 addrspace(4)* %dispatch_ptr.bc

  %queue_ptr = call noalias i8 addrspace(4)* @llvm.amdgcn.queue.ptr() #0
  %queue_ptr.bc = bitcast i8 addrspace(4)* %queue_ptr to i32 addrspace(4)*
  %val1 = load volatile i32, i32 addrspace(4)* %queue_ptr.bc

  %implicitarg.ptr = call noalias i8 addrspace(4)* @llvm.amdgcn.implicitarg.ptr() #0
  %implicitarg.ptr.bc = bitcast i8 addrspace(4)* %implicitarg.ptr to i32 addrspace(4)*
  %val2 = load volatile i32, i32 addrspace(4)* %implicitarg.ptr.bc

  %val3 = call i64 @llvm.amdgcn.dispatch.id()
  call void asm sideeffect "; use $0", "s"(i64 %val3)

  %val4 = call i32 @llvm.amdgcn.workgroup.id.x()
  call void asm sideeffect "; use $0", "s"(i32 %val4)

  %val5 = call i32 @llvm.amdgcn.workgroup.id.y()
  call void asm sideeffect "; use $0", "s"(i32 %val5)

  %val6 = call i32 @llvm.amdgcn.workgroup.id.z()
  call void asm sideeffect "; use $0", "s"(i32 %val6)

  ret void
}

declare i32 @llvm.amdgcn.workgroup.id.x() #0
declare i32 @llvm.amdgcn.workgroup.id.y() #0
declare i32 @llvm.amdgcn.workgroup.id.z() #0
declare noalias i8 addrspace(4)* @llvm.amdgcn.queue.ptr() #0
declare noalias i8 addrspace(4)* @llvm.amdgcn.kernarg.segment.ptr() #0
declare noalias i8 addrspace(4)* @llvm.amdgcn.implicitarg.ptr() #0
declare i64 @llvm.amdgcn.dispatch.id() #0
declare noalias i8 addrspace(4)* @llvm.amdgcn.dispatch.ptr() #0

attributes #0 = { nounwind readnone speculatable }
attributes #1 = { nounwind noinline }
