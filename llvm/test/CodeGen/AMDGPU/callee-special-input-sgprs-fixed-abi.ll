; RUN: opt -passes=amdgpu-attributor -mcpu=kaveri < %s | llc -enable-ipra=0 | FileCheck -enable-var-scope -check-prefixes=GCN,CIVI %s
; RUN: opt -passes=amdgpu-attributor -mcpu=gfx900 < %s | llc -enable-ipra=0 | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9 %s

target triple = "amdgcn-amd-amdhsa"

; GCN-LABEL: {{^}}use_dispatch_ptr:
; GCN: s_load_dword s{{[0-9]+}}, s[4:5]
define hidden void @use_dispatch_ptr() #1 {
  %dispatch_ptr = call noalias ptr addrspace(4) @llvm.amdgcn.dispatch.ptr() #0
  %value = load volatile i32, ptr addrspace(4) %dispatch_ptr
  ret void
}

; GCN-LABEL: {{^}}kern_indirect_use_dispatch_ptr:
; GCN-NOT: s[4:5]
; GCN-NOT: s4
; GCN-NOT: s5
; GCN: .amdhsa_user_sgpr_dispatch_ptr 1
define amdgpu_kernel void @kern_indirect_use_dispatch_ptr(i32) #1 {
  call void @use_dispatch_ptr()
  ret void
}

; GCN-LABEL: {{^}}use_queue_ptr:
; GCN: s_load_dword s{{[0-9]+}}, s[6:7]
define hidden void @use_queue_ptr() #1 {
  %queue_ptr = call noalias ptr addrspace(4) @llvm.amdgcn.queue.ptr() #0
  %value = load volatile i32, ptr addrspace(4) %queue_ptr
  ret void
}

; GCN-LABEL: {{^}}kern_indirect_use_queue_ptr:
; GCN: s_swappc_b64 s[30:31], s[10:11]
; GCN: .amdhsa_user_sgpr_queue_ptr 1
define amdgpu_kernel void @kern_indirect_use_queue_ptr(i32) #1 {
  call void @use_queue_ptr()
  ret void
}

; GCN-LABEL: {{^}}use_queue_ptr_addrspacecast:
; CIVI: s_load_dword [[APERTURE_LOAD:s[0-9]+]], s[4:5], 0x0
; CIVI: v_mov_b32_e32 v[[LO:[0-9]+]], 16
; CIVI-DAG: v_mov_b32_e32 v[[HI:[0-9]+]], [[APERTURE_LOAD]]

; GFX9: s_mov_b64 s[{{[0-9]+}}:[[HI:[0-9]+]]], src_shared_base
; GFX9-DAG: v_mov_b32_e32 v[[VGPR_HI:[0-9]+]], s[[HI]]
; GFX9: {{flat|global}}_store_dword v{{\[[0-9]+}}:[[VGPR_HI]]]

; CIVI: {{flat|global}}_store_dword v[[[LO]]:[[HI]]]
define hidden void @use_queue_ptr_addrspacecast() #1 {
  %asc = addrspacecast ptr addrspace(3) inttoptr (i32 16 to ptr addrspace(3)) to ptr
  store volatile i32 0, ptr %asc
  ret void
}

; GCN-LABEL: {{^}}kern_indirect_use_queue_ptr_addrspacecast:
; CIVI: s_swappc_b64 s[30:31], s[4:5]
; CIVI: .amdhsa_user_sgpr_queue_ptr 0

; GFX9-NOT: s_mov_b64 s[6:7]
; GFX9: .amdhsa_user_sgpr_queue_ptr 0
define amdgpu_kernel void @kern_indirect_use_queue_ptr_addrspacecast(i32) #1 {
  call void @use_queue_ptr_addrspacecast()
  ret void
}

; Not really supported in callable functions.
; GCN-LABEL: {{^}}use_kernarg_segment_ptr:
; GCN: s_mov_b64 [[PTR:s\[[0-9]+:[0-9]+\]]], 0
; GCN: s_load_dword s{{[0-9]+}}, [[PTR]], 0x0
define hidden void @use_kernarg_segment_ptr() #1 {
  %kernarg_segment_ptr = call noalias ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr() #0
  %value = load volatile i32, ptr addrspace(4) %kernarg_segment_ptr
  ret void
}

; GCN-LABEL: {{^}}use_implicitarg_ptr:
; GCN: s_load_dword s{{[0-9]+}}, s[8:9]
define hidden void @use_implicitarg_ptr() #1 {
  %implicit.arg.ptr = call noalias ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr() #0
  %value = load volatile i32, ptr addrspace(4) %implicit.arg.ptr
  ret void
}

; GCN-LABEL: {{^}}kern_indirect_use_kernarg_segment_ptr:
; GCN: .amdhsa_user_sgpr_kernarg_segment_ptr 1
define amdgpu_kernel void @kern_indirect_use_kernarg_segment_ptr(i32) #1 {
  call void @use_kernarg_segment_ptr()
  ret void
}

; GCN-LABEL: {{^}}use_dispatch_id:
; GCN: ; use s[10:11]
define hidden void @use_dispatch_id() #1 {
  %id = call i64 @llvm.amdgcn.dispatch.id()
  call void asm sideeffect "; use $0", "s"(i64 %id)
  ret void
}

; No kernarg segment so that there is a mov to check. With kernarg
; pointer enabled, it happens to end up in the right place anyway.

; GCN-LABEL: {{^}}kern_indirect_use_dispatch_id:
; GCN: s_mov_b64 s[10:11], s[4:5]
; GCN: .amdhsa_user_sgpr_dispatch_id 1
define amdgpu_kernel void @kern_indirect_use_dispatch_id() #1 {
  call void @use_dispatch_id()
  ret void
}

; GCN-LABEL: {{^}}use_workgroup_id_x:
; GCN: s_waitcnt
; GCN: ; use s12
define hidden void @use_workgroup_id_x() #1 {
  %val = call i32 @llvm.amdgcn.workgroup.id.x()
  call void asm sideeffect "; use $0", "s"(i32 %val)
  ret void
}

; GCN-LABEL: {{^}}use_stack_workgroup_id_x:
; GCN: s_waitcnt
; GCN-NOT: s32
; GCN: buffer_store_dword v0, off, s[0:3], s32{{$}}
; GCN: ; use s12
; GCN: s_setpc_b64
define hidden void @use_stack_workgroup_id_x() #1 {
  %alloca = alloca i32, addrspace(5)
  store volatile i32 0, ptr addrspace(5) %alloca
  %val = call i32 @llvm.amdgcn.workgroup.id.x()
  call void asm sideeffect "; use $0", "s"(i32 %val)
  ret void
}

; GCN-LABEL: {{^}}use_workgroup_id_y:
; GCN: s_waitcnt
; GCN: ; use s13
define hidden void @use_workgroup_id_y() #1 {
  %val = call i32 @llvm.amdgcn.workgroup.id.y()
  call void asm sideeffect "; use $0", "s"(i32 %val)
  ret void
}

; GCN-LABEL: {{^}}use_workgroup_id_z:
; GCN: s_waitcnt
; GCN: ; use s14
define hidden void @use_workgroup_id_z() #1 {
  %val = call i32 @llvm.amdgcn.workgroup.id.z()
  call void asm sideeffect "; use $0", "s"(i32 %val)
  ret void
}

; GCN-LABEL: {{^}}use_workgroup_id_xy:
; GCN: ; use s12
; GCN: ; use s13
define hidden void @use_workgroup_id_xy() #1 {
  %val0 = call i32 @llvm.amdgcn.workgroup.id.x()
  %val1 = call i32 @llvm.amdgcn.workgroup.id.y()
  call void asm sideeffect "; use $0", "s"(i32 %val0)
  call void asm sideeffect "; use $0", "s"(i32 %val1)
  ret void
}

; GCN-LABEL: {{^}}use_workgroup_id_xyz:
; GCN: ; use s12
; GCN: ; use s13
; GCN: ; use s14
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
; GCN: ; use s12
; GCN: ; use s14
define hidden void @use_workgroup_id_xz() #1 {
  %val0 = call i32 @llvm.amdgcn.workgroup.id.x()
  %val1 = call i32 @llvm.amdgcn.workgroup.id.z()
  call void asm sideeffect "; use $0", "s"(i32 %val0)
  call void asm sideeffect "; use $0", "s"(i32 %val1)
  ret void
}

; GCN-LABEL: {{^}}use_workgroup_id_yz:
; GCN: ; use s13
; GCN: ; use s14
define hidden void @use_workgroup_id_yz() #1 {
  %val0 = call i32 @llvm.amdgcn.workgroup.id.y()
  %val1 = call i32 @llvm.amdgcn.workgroup.id.z()
  call void asm sideeffect "; use $0", "s"(i32 %val0)
  call void asm sideeffect "; use $0", "s"(i32 %val1)
  ret void
}

; GCN-LABEL: {{^}}kern_indirect_use_workgroup_id_x:
; GCN-NOT: s6
; GCN: s_getpc_b64 s[4:5]
; GCN-NEXT: s_add_u32 s4, s4, use_workgroup_id_x@rel32@lo+4
; GCN-NEXT: s_addc_u32 s5, s5, use_workgroup_id_x@rel32@hi+12
; GCN-NOT: s6
; GCN: s_mov_b32 s12, s6
; GCN: s_mov_b32 s32, 0
; GCN: s_swappc_b64
; GCN-NEXT: s_endpgm

; GCN: .amdhsa_system_sgpr_workgroup_id_x 1
; GCN: .amdhsa_system_sgpr_workgroup_id_y 0
; GCN: .amdhsa_system_sgpr_workgroup_id_z 0
define amdgpu_kernel void @kern_indirect_use_workgroup_id_x() #1 {
  call void @use_workgroup_id_x()
  ret void
}

; GCN-LABEL: {{^}}kern_indirect_use_workgroup_id_y:
; GCN-NOT: s12
; GCN: s_mov_b32 s13, s7
; GCN-NOT: s12
; GCN: s_mov_b32 s32, 0
; GCN: s_swappc_b64

; GCN: .amdhsa_system_sgpr_workgroup_id_x 1
; GCN: .amdhsa_system_sgpr_workgroup_id_y 1
; GCN: .amdhsa_system_sgpr_workgroup_id_z 0
define amdgpu_kernel void @kern_indirect_use_workgroup_id_y() #1 {
  call void @use_workgroup_id_y()
  ret void
}

; GCN-LABEL: {{^}}kern_indirect_use_workgroup_id_z:
; GCN-NOT: s12
; GCN-NOT: s13
; GCN: s_mov_b32 s14, s7
; GCN-NOT: s12
; GCN-NOT: s13

; GCN: s_mov_b32 s32, 0
; GCN: s_swappc_b64

; GCN: .amdhsa_system_sgpr_workgroup_id_x 1
; GCN: .amdhsa_system_sgpr_workgroup_id_y 0
; GCN: .amdhsa_system_sgpr_workgroup_id_z 1
define amdgpu_kernel void @kern_indirect_use_workgroup_id_z() #1 {
  call void @use_workgroup_id_z()
  ret void
}

; GCN-LABEL: {{^}}kern_indirect_use_workgroup_id_xy:
; GCN-NOT: s14
; GCN: s_mov_b32 s12, s6
; GCN-NEXT: s_mov_b32 s13, s7
; GCN-NOT: s14

; GCN: s_mov_b32 s32, 0
; GCN: s_swappc_b64

; GCN: .amdhsa_system_sgpr_workgroup_id_x 1
; GCN: .amdhsa_system_sgpr_workgroup_id_y 1
; GCN: .amdhsa_system_sgpr_workgroup_id_z 0
define amdgpu_kernel void @kern_indirect_use_workgroup_id_xy() #1 {
  call void @use_workgroup_id_xy()
  ret void
}

; GCN-LABEL: {{^}}kern_indirect_use_workgroup_id_xyz:
; GCN: s_mov_b32 s12, s6
; GCN: s_mov_b32 s13, s7
; GCN: s_mov_b32 s14, s8
; GCN: s_mov_b32 s32, 0
; GCN: s_swappc_b64

; GCN: .amdhsa_system_sgpr_workgroup_id_x 1
; GCN: .amdhsa_system_sgpr_workgroup_id_y 1
; GCN: .amdhsa_system_sgpr_workgroup_id_z 1
define amdgpu_kernel void @kern_indirect_use_workgroup_id_xyz() #1 {
  call void @use_workgroup_id_xyz()
  ret void
}

; GCN-LABEL: {{^}}kern_indirect_use_workgroup_id_xz:

; GCN-NOT: s13
; GCN: s_mov_b32 s12, s6
; GCN-NEXT: s_mov_b32 s14, s7
; GCN-NOT: s13

; GCN: s_mov_b32 s32, 0
; GCN: s_swappc_b64

; GCN: .amdhsa_system_sgpr_workgroup_id_x 1
; GCN: .amdhsa_system_sgpr_workgroup_id_y 0
; GCN: .amdhsa_system_sgpr_workgroup_id_z 1
define amdgpu_kernel void @kern_indirect_use_workgroup_id_xz() #1 {
  call void @use_workgroup_id_xz()
  ret void
}

; GCN-LABEL: {{^}}kern_indirect_use_workgroup_id_yz:

; GCN: s_mov_b32 s13, s7
; GCN: s_mov_b32 s14, s8

; GCN: s_mov_b32 s32, 0
; GCN: s_swappc_b64

; GCN: .amdhsa_system_sgpr_workgroup_id_x 1
; GCN: .amdhsa_system_sgpr_workgroup_id_y 1
; GCN: .amdhsa_system_sgpr_workgroup_id_z 1
define amdgpu_kernel void @kern_indirect_use_workgroup_id_yz() #1 {
  call void @use_workgroup_id_yz()
  ret void
}

; Argument is in right place already
; GCN-LABEL: {{^}}func_indirect_use_workgroup_id_x:
; GCN-NOT: s12
; GCN-NOT: s13
; GCN-NOT: s14
; GCN: v_readlane_b32 s30, v40, 0
define hidden void @func_indirect_use_workgroup_id_x() #1 {
  call void @use_workgroup_id_x()
  ret void
}

; Argument is in right place already. We are free to clobber other
; SGPR arguments
; GCN-LABEL: {{^}}func_indirect_use_workgroup_id_y:
; GCN-NOT: s12
; GCN-NOT: s13
; GCN-NOT: s14
define hidden void @func_indirect_use_workgroup_id_y() #1 {
  call void @use_workgroup_id_y()
  ret void
}

; GCN-LABEL: {{^}}func_indirect_use_workgroup_id_z:
; GCN-NOT: s12
; GCN-NOT: s13
; GCN-NOT: s14
define hidden void @func_indirect_use_workgroup_id_z() #1 {
  call void @use_workgroup_id_z()
  ret void
}

; GCN-LABEL: {{^}}other_arg_use_workgroup_id_x:
; CIVI: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, v0
; GFX9: global_store_dword v{{\[[0-9]+:[0-9]+\]}}, v0, off
; GCN: ; use s12
define hidden void @other_arg_use_workgroup_id_x(i32 %arg0) #1 {
  %val = call i32 @llvm.amdgcn.workgroup.id.x()
  store volatile i32 %arg0, ptr addrspace(1) undef
  call void asm sideeffect "; use $0", "s"(i32 %val)
  ret void
}

; GCN-LABEL: {{^}}other_arg_use_workgroup_id_y:
; CIVI: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, v0
; GFX9: global_store_dword v{{\[[0-9]+:[0-9]+\]}}, v0, off
; GCN: ; use s13
define hidden void @other_arg_use_workgroup_id_y(i32 %arg0) #1 {
  %val = call i32 @llvm.amdgcn.workgroup.id.y()
  store volatile i32 %arg0, ptr addrspace(1) undef
  call void asm sideeffect "; use $0", "s"(i32 %val)
  ret void
}

; GCN-LABEL: {{^}}other_arg_use_workgroup_id_z:
; CIVI: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, v0
; GFX9: global_store_dword v{{\[[0-9]+:[0-9]+\]}}, v0, off
; GCN: ; use s14
define hidden void @other_arg_use_workgroup_id_z(i32 %arg0) #1 {
  %val = call i32 @llvm.amdgcn.workgroup.id.z()
  store volatile i32 %arg0, ptr addrspace(1) undef
  call void asm sideeffect "; use $0", "s"(i32 %val)
  ret void
}

; GCN-LABEL: {{^}}kern_indirect_other_arg_use_workgroup_id_x:

; GCN-NOT: s13
; GCN-NOT: s14
; GCN-DAG: s_mov_b32 s12, s6
; GCN-DAG: v_mov_b32_e32 v0, 0x22b
; GCN-NOT: s13
; GCN-NOT: s14

; GCN-DAG: s_mov_b32 s32, 0
; GCN: s_swappc_b64

; GCN: .amdhsa_system_sgpr_workgroup_id_x 1
; GCN: .amdhsa_system_sgpr_workgroup_id_y 0
; GCN: .amdhsa_system_sgpr_workgroup_id_z 0
define amdgpu_kernel void @kern_indirect_other_arg_use_workgroup_id_x() #1 {
  call void @other_arg_use_workgroup_id_x(i32 555)
  ret void
}

; GCN-LABEL: {{^}}kern_indirect_other_arg_use_workgroup_id_y:
; GCN-DAG: v_mov_b32_e32 v0, 0x22b
; GCN-DAG: s_mov_b32 s13, s7

; GCN-DAG: s_mov_b32 s32, 0
; GCN: s_swappc_b64

; GCN: .amdhsa_system_sgpr_workgroup_id_x 1
; GCN: .amdhsa_system_sgpr_workgroup_id_y 1
; GCN: .amdhsa_system_sgpr_workgroup_id_z 0
define amdgpu_kernel void @kern_indirect_other_arg_use_workgroup_id_y() #1 {
  call void @other_arg_use_workgroup_id_y(i32 555)
  ret void
}

; GCN-LABEL: {{^}}kern_indirect_other_arg_use_workgroup_id_z:
; GCN-DAG: v_mov_b32_e32 v0, 0x22b
; GCN-DAG: s_mov_b32 s14, s7

; GCN: s_mov_b32 s32, 0
; GCN: s_swappc_b64

; GCN: .amdhsa_system_sgpr_workgroup_id_x 1
; GCN: .amdhsa_system_sgpr_workgroup_id_y 0
; GCN: .amdhsa_system_sgpr_workgroup_id_z 1
define amdgpu_kernel void @kern_indirect_other_arg_use_workgroup_id_z() #1 {
  call void @other_arg_use_workgroup_id_z(i32 555)
  ret void
}

; GCN-LABEL: {{^}}use_every_sgpr_input:
; GCN: buffer_store_dword v{{[0-9]+}}, off, s[0:3], s32{{$}}
; GCN: s_load_dword s{{[0-9]+}}, s[4:5]
; GCN: s_load_dword s{{[0-9]+}}, s[6:7]
; GCN: s_load_dword s{{[0-9]+}}, s[8:9]
; GCN: ; use s[10:11]
; GCN: ; use s12
; GCN: ; use s13
; GCN: ; use s14
define hidden void @use_every_sgpr_input() #1 {
  %alloca = alloca i32, align 4, addrspace(5)
  store volatile i32 0, ptr addrspace(5) %alloca

  %dispatch_ptr = call noalias ptr addrspace(4) @llvm.amdgcn.dispatch.ptr() #0
  %val0 = load volatile i32, ptr addrspace(4) %dispatch_ptr

  %queue_ptr = call noalias ptr addrspace(4) @llvm.amdgcn.queue.ptr() #0
  %val1 = load volatile i32, ptr addrspace(4) %queue_ptr

  %implicitarg.ptr = call noalias ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr() #0
  %val2 = load volatile i32, ptr addrspace(4) %implicitarg.ptr

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
; GCN: s_mov_b32 s32, 0
; GCN: s_swappc_b64

; GCN: .amdhsa_user_sgpr_private_segment_buffer 1
; GCN: .amdhsa_user_sgpr_dispatch_ptr 1
; GCN: .amdhsa_user_sgpr_queue_ptr 1
; GCN: .amdhsa_user_sgpr_kernarg_segment_ptr 1
; GCN: .amdhsa_user_sgpr_dispatch_id 1
; GCN: .amdhsa_user_sgpr_flat_scratch_init 1
; GCN: .amdhsa_user_sgpr_private_segment_size 0
; GCN: .amdhsa_system_sgpr_private_segment_wavefront_offset 1
; GCN: .amdhsa_system_sgpr_workgroup_id_x 1
; GCN: .amdhsa_system_sgpr_workgroup_id_y 1
; GCN: .amdhsa_system_sgpr_workgroup_id_z 1
; GCN: .amdhsa_system_sgpr_workgroup_info 0
; GCN: .amdhsa_system_vgpr_workitem_id 0
define amdgpu_kernel void @kern_indirect_use_every_sgpr_input(i8) #1 {
  call void @use_every_sgpr_input()
  ret void
}

; We have to pass the kernarg segment, but there are no kernel
; arguments so null is passed.
; GCN-LABEL: {{^}}kern_indirect_use_every_sgpr_input_no_kernargs:
; GCN: s_mov_b64 s[10:11], s[8:9]
; GCN: s_mov_b32 s32, 0
; GCN: s_swappc_b64

; GCN: .amdhsa_user_sgpr_private_segment_buffer 1
; GCN: .amdhsa_user_sgpr_dispatch_ptr 1
; GCN: .amdhsa_user_sgpr_queue_ptr 1
; GCN: .amdhsa_user_sgpr_kernarg_segment_ptr 0
; GCN: .amdhsa_user_sgpr_dispatch_id 1
; GCN: .amdhsa_user_sgpr_flat_scratch_init 1
; GCN: .amdhsa_user_sgpr_private_segment_size 0
; GCN: .amdhsa_system_sgpr_private_segment_wavefront_offset 1
; GCN: .amdhsa_system_sgpr_workgroup_id_x 1
; GCN: .amdhsa_system_sgpr_workgroup_id_y 1
; GCN: .amdhsa_system_sgpr_workgroup_id_z 1
; GCN: .amdhsa_system_sgpr_workgroup_info 0
; GCN: .amdhsa_system_vgpr_workitem_id 0
define amdgpu_kernel void @kern_indirect_use_every_sgpr_input_no_kernargs() #2 {
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
; GCN-NOT: s14
; GCN: s_or_saveexec_b64 s[16:17], -1
define hidden void @func_indirect_use_every_sgpr_input() #1 {
  call void @use_every_sgpr_input()
  ret void
}

; GCN-LABEL: {{^}}func_use_every_sgpr_input_call_use_workgroup_id_xyz:
; GCN-NOT: s12
; GCN-NOT: s13
; GCN-NOT: s14
; GCN: ; use s[10:11]
; GCN: ; use s12
; GCN: ; use s13
; GCN: ; use s14

; GCN: s_swappc_b64
define hidden void @func_use_every_sgpr_input_call_use_workgroup_id_xyz() #1 {
  %alloca = alloca i32, align 4, addrspace(5)
  store volatile i32 0, ptr addrspace(5) %alloca

  %dispatch_ptr = call noalias ptr addrspace(4) @llvm.amdgcn.dispatch.ptr() #0
  %val0 = load volatile i32, ptr addrspace(4) %dispatch_ptr

  %queue_ptr = call noalias ptr addrspace(4) @llvm.amdgcn.queue.ptr() #0
  %val1 = load volatile i32, ptr addrspace(4) %queue_ptr

  %kernarg_segment_ptr = call noalias ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr() #0
  %val2 = load volatile i32, ptr addrspace(4) %kernarg_segment_ptr

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

declare i32 @llvm.amdgcn.workgroup.id.x() #0
declare i32 @llvm.amdgcn.workgroup.id.y() #0
declare i32 @llvm.amdgcn.workgroup.id.z() #0
declare noalias ptr addrspace(4) @llvm.amdgcn.queue.ptr() #0
declare noalias ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr() #0
declare noalias ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr() #0
declare i64 @llvm.amdgcn.dispatch.id() #0
declare noalias ptr addrspace(4) @llvm.amdgcn.dispatch.ptr() #0

attributes #0 = { nounwind readnone speculatable }
attributes #1 = { nounwind noinline }
attributes #2 = { nounwind noinline "amdgpu-implicitarg-num-bytes"="0" }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 500}
