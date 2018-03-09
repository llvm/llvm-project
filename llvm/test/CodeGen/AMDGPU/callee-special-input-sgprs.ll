; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=kaveri -enable-ipra=0 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,CIVI %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -enable-ipra=0 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9 %s

; GCN-LABEL: {{^}}use_dispatch_ptr:
; GCN: s_load_dword s{{[0-9]+}}, s[6:7], 0x0
define void @use_dispatch_ptr() #1 {
  %dispatch_ptr = call noalias i8 addrspace(2)* @llvm.amdgcn.dispatch.ptr() #0
  %header_ptr = bitcast i8 addrspace(2)* %dispatch_ptr to i32 addrspace(2)*
  %value = load volatile i32, i32 addrspace(2)* %header_ptr
  ret void
}

; GCN-LABEL: {{^}}kern_indirect_use_dispatch_ptr:
; GCN: enable_sgpr_dispatch_ptr = 1
; GCN: s_mov_b64 s[6:7], s[4:5]
define amdgpu_kernel void @kern_indirect_use_dispatch_ptr(i32) #1 {
  call void @use_dispatch_ptr()
  ret void
}

; GCN-LABEL: {{^}}use_queue_ptr:
; GCN: s_load_dword s{{[0-9]+}}, s[6:7], 0x0
define void @use_queue_ptr() #1 {
  %queue_ptr = call noalias i8 addrspace(2)* @llvm.amdgcn.queue.ptr() #0
  %header_ptr = bitcast i8 addrspace(2)* %queue_ptr to i32 addrspace(2)*
  %value = load volatile i32, i32 addrspace(2)* %header_ptr
  ret void
}

; GCN-LABEL: {{^}}kern_indirect_use_queue_ptr:
; GCN: enable_sgpr_queue_ptr = 1
; GCN: s_mov_b64 s[6:7], s[4:5]
; GCN: s_swappc_b64
define amdgpu_kernel void @kern_indirect_use_queue_ptr(i32) #1 {
  call void @use_queue_ptr()
  ret void
}

; GCN-LABEL: {{^}}use_queue_ptr_addrspacecast:
; CIVI: s_load_dword [[APERTURE_LOAD:s[0-9]+]], s[6:7], 0x10
; GFX9: s_getreg_b32 [[APERTURE_LOAD:s[0-9]+]]

; GCN: v_mov_b32_e32 v[[HI:[0-9]+]], [[APERTURE_LOAD]]
; GCN: {{flat|global}}_store_dword v{{\[[0-9]+}}:[[HI]]{{\]}}
define void @use_queue_ptr_addrspacecast() #1 {
  %asc = addrspacecast i32 addrspace(3)* inttoptr (i32 16 to i32 addrspace(3)*) to i32 addrspace(4)*
  store volatile i32 0, i32 addrspace(4)* %asc
  ret void
}

; GCN-LABEL: {{^}}kern_indirect_use_queue_ptr_addrspacecast:
; CIVI: enable_sgpr_queue_ptr = 1

; CIVI: s_mov_b64 s[6:7], s[4:5]
; GFX9-NOT: s_mov_b64
; GCN: s_swappc_b64
define amdgpu_kernel void @kern_indirect_use_queue_ptr_addrspacecast(i32) #1 {
  call void @use_queue_ptr_addrspacecast()
  ret void
}

; GCN-LABEL: {{^}}use_kernarg_segment_ptr:
; GCN: s_load_dword s{{[0-9]+}}, s[6:7], 0x0
define void @use_kernarg_segment_ptr() #1 {
  %kernarg_segment_ptr = call noalias i8 addrspace(2)* @llvm.amdgcn.kernarg.segment.ptr() #0
  %header_ptr = bitcast i8 addrspace(2)* %kernarg_segment_ptr to i32 addrspace(2)*
  %value = load volatile i32, i32 addrspace(2)* %header_ptr
  ret void
}

; GCN-LABEL: {{^}}kern_indirect_use_kernarg_segment_ptr:
; GCN: enable_sgpr_kernarg_segment_ptr = 1
; GCN: s_mov_b64 s[6:7], s[4:5]
; GCN: s_swappc_b64
define amdgpu_kernel void @kern_indirect_use_kernarg_segment_ptr(i32) #1 {
  call void @use_kernarg_segment_ptr()
  ret void
}

; GCN-LABEL: {{^}}use_dispatch_id:
; GCN: ; use s[6:7]
define void @use_dispatch_id() #1 {
  %id = call i64 @llvm.amdgcn.dispatch.id()
  call void asm sideeffect "; use $0", "s"(i64 %id)
  ret void
}

; No kernarg segment so that there is a mov to check. With kernarg
; pointer enabled, it happens to end up in the right place anyway.

; GCN-LABEL: {{^}}kern_indirect_use_dispatch_id:
; GCN: enable_sgpr_dispatch_id = 1

; GCN: s_mov_b64 s[6:7], s[4:5]
define amdgpu_kernel void @kern_indirect_use_dispatch_id() #1 {
  call void @use_dispatch_id()
  ret void
}

; GCN-LABEL: {{^}}use_workgroup_id_x:
; GCN: s_waitcnt
; GCN: ; use s6
define void @use_workgroup_id_x() #1 {
  %val = call i32 @llvm.amdgcn.workgroup.id.x()
  call void asm sideeffect "; use $0", "s"(i32 %val)
  ret void
}

; GCN-LABEL: {{^}}use_stack_workgroup_id_x:
; GCN: s_waitcnt
; GCN: s_mov_b32 s5, s32
; GCN: buffer_store_dword v0, off, s[0:3], s5 offset:4
; GCN: ; use s6
; GCN: s_setpc_b64
define void @use_stack_workgroup_id_x() #1 {
  %alloca = alloca i32
  store volatile i32 0, i32* %alloca
  %val = call i32 @llvm.amdgcn.workgroup.id.x()
  call void asm sideeffect "; use $0", "s"(i32 %val)
  ret void
}

; GCN-LABEL: {{^}}use_workgroup_id_y:
; GCN: s_waitcnt
; GCN: ; use s6
define void @use_workgroup_id_y() #1 {
  %val = call i32 @llvm.amdgcn.workgroup.id.y()
  call void asm sideeffect "; use $0", "s"(i32 %val)
  ret void
}

; GCN-LABEL: {{^}}use_workgroup_id_z:
; GCN: s_waitcnt
; GCN: ; use s6
define void @use_workgroup_id_z() #1 {
  %val = call i32 @llvm.amdgcn.workgroup.id.z()
  call void asm sideeffect "; use $0", "s"(i32 %val)
  ret void
}

; GCN-LABEL: {{^}}use_workgroup_id_xy:
; GCN: ; use s6
; GCN: ; use s7
define void @use_workgroup_id_xy() #1 {
  %val0 = call i32 @llvm.amdgcn.workgroup.id.x()
  %val1 = call i32 @llvm.amdgcn.workgroup.id.y()
  call void asm sideeffect "; use $0", "s"(i32 %val0)
  call void asm sideeffect "; use $0", "s"(i32 %val1)
  ret void
}

; GCN-LABEL: {{^}}use_workgroup_id_xyz:
; GCN: ; use s6
; GCN: ; use s7
; GCN: ; use s8
define void @use_workgroup_id_xyz() #1 {
  %val0 = call i32 @llvm.amdgcn.workgroup.id.x()
  %val1 = call i32 @llvm.amdgcn.workgroup.id.y()
  %val2 = call i32 @llvm.amdgcn.workgroup.id.z()
  call void asm sideeffect "; use $0", "s"(i32 %val0)
  call void asm sideeffect "; use $0", "s"(i32 %val1)
  call void asm sideeffect "; use $0", "s"(i32 %val2)
  ret void
}

; GCN-LABEL: {{^}}use_workgroup_id_xz:
; GCN: ; use s6
; GCN: ; use s7
define void @use_workgroup_id_xz() #1 {
  %val0 = call i32 @llvm.amdgcn.workgroup.id.x()
  %val1 = call i32 @llvm.amdgcn.workgroup.id.z()
  call void asm sideeffect "; use $0", "s"(i32 %val0)
  call void asm sideeffect "; use $0", "s"(i32 %val1)
  ret void
}

; GCN-LABEL: {{^}}use_workgroup_id_yz:
; GCN: ; use s6
; GCN: ; use s7
define void @use_workgroup_id_yz() #1 {
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
; GCN: s_mov_b32 s33, s7
; GCN-NOT: s6
; GCN: s_mov_b32 s4, s33
; GCN-NOT: s6
; GCN: s_mov_b32 s32, s33
; GCN: s_swappc_b64
define amdgpu_kernel void @kern_indirect_use_workgroup_id_x() #1 {
  call void @use_workgroup_id_x()
  ret void
}

; GCN-LABEL: {{^}}kern_indirect_use_workgroup_id_y:
; GCN: enable_sgpr_workgroup_id_x = 1
; GCN: enable_sgpr_workgroup_id_y = 1
; GCN: enable_sgpr_workgroup_id_z = 0

; GCN: s_mov_b32 s33, s8
; GCN: s_mov_b32 s4, s33
; GCN: s_mov_b32 s6, s7
; GCN: s_mov_b32 s32, s33
; GCN: s_swappc_b64
define amdgpu_kernel void @kern_indirect_use_workgroup_id_y() #1 {
  call void @use_workgroup_id_y()
  ret void
}

; GCN-LABEL: {{^}}kern_indirect_use_workgroup_id_z:
; GCN: enable_sgpr_workgroup_id_x = 1
; GCN: enable_sgpr_workgroup_id_y = 0
; GCN: enable_sgpr_workgroup_id_z = 1

; GCN: s_mov_b32 s33, s8
; GCN: s_mov_b32 s4, s33
; GCN: s_mov_b32 s6, s7
; GCN: s_swappc_b64
define amdgpu_kernel void @kern_indirect_use_workgroup_id_z() #1 {
  call void @use_workgroup_id_z()
  ret void
}

; GCN-LABEL: {{^}}kern_indirect_use_workgroup_id_xy:
; GCN: enable_sgpr_workgroup_id_x = 1
; GCN: enable_sgpr_workgroup_id_y = 1
; GCN: enable_sgpr_workgroup_id_z = 0

; GCN: s_mov_b32 s33, s8
; GCN-NOT: s6
; GCN-NOT: s7
; GCN: s_mov_b32 s4, s33
; GCN-NOT: s6
; GCN-NOT: s7
; GCN: s_mov_b32 s32, s33
; GCN-NOT: s6
; GCN-NOT: s7
; GCN: s_swappc_b64
define amdgpu_kernel void @kern_indirect_use_workgroup_id_xy() #1 {
  call void @use_workgroup_id_xy()
  ret void
}

; GCN-LABEL: {{^}}kern_indirect_use_workgroup_id_xyz:
; GCN: enable_sgpr_workgroup_id_x = 1
; GCN: enable_sgpr_workgroup_id_y = 1
; GCN: enable_sgpr_workgroup_id_z = 1

; GCN: s_mov_b32 s33, s9

; GCN-NOT: s6
; GCN-NOT: s7
; GCN-NOT: s8

; GCN: s_mov_b32 s4, s33

; GCN-NOT: s6
; GCN-NOT: s7
; GCN-NOT: s8

; GCN: s_mov_b32 s32, s33

; GCN-NOT: s6
; GCN-NOT: s7
; GCN-NOT: s8

; GCN: s_swappc_b64
define amdgpu_kernel void @kern_indirect_use_workgroup_id_xyz() #1 {
  call void @use_workgroup_id_xyz()
  ret void
}

; GCN-LABEL: {{^}}kern_indirect_use_workgroup_id_xz:
; GCN: enable_sgpr_workgroup_id_x = 1
; GCN: enable_sgpr_workgroup_id_y = 0
; GCN: enable_sgpr_workgroup_id_z = 1

; GCN: s_mov_b32 s33, s8
; GCN-NOT: s6
; GCN-NOT: s7

; GCN: s_mov_b32 s4, s33
; GCN-NOT: s6
; GCN-NOT: s7

; GCN: s_mov_b32 s32, s33
; GCN-NOT: s6
; GCN-NOT: s7

; GCN: s_swappc_b64
define amdgpu_kernel void @kern_indirect_use_workgroup_id_xz() #1 {
  call void @use_workgroup_id_xz()
  ret void
}

; GCN-LABEL: {{^}}kern_indirect_use_workgroup_id_yz:
; GCN: enable_sgpr_workgroup_id_x = 1
; GCN: enable_sgpr_workgroup_id_y = 1
; GCN: enable_sgpr_workgroup_id_z = 1

; GCN: s_mov_b32 s33, s9
; GCN: s_mov_b32 s6, s7
; GCN: s_mov_b32 s4, s33
; GCN: s_mov_b32 s7, s8
; GCN: s_mov_b32 s32, s33
; GCN: s_swappc_b64
define amdgpu_kernel void @kern_indirect_use_workgroup_id_yz() #1 {
  call void @use_workgroup_id_yz()
  ret void
}

; Argument is in right place already
; GCN-LABEL: {{^}}func_indirect_use_workgroup_id_x:
; GCN-NOT: s6
define void @func_indirect_use_workgroup_id_x() #1 {
  call void @use_workgroup_id_x()
  ret void
}

; GCN-LABEL: {{^}}func_indirect_use_workgroup_id_y:
; GCN-NOT: s6
define void @func_indirect_use_workgroup_id_y() #1 {
  call void @use_workgroup_id_y()
  ret void
}

; GCN-LABEL: {{^}}func_indirect_use_workgroup_id_z:
; GCN-NOT: s6
define void @func_indirect_use_workgroup_id_z() #1 {
  call void @use_workgroup_id_z()
  ret void
}

; GCN-LABEL: {{^}}other_arg_use_workgroup_id_x:
; GCN: {{flat|global}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, v0
; GCN: ; use s6
define void @other_arg_use_workgroup_id_x(i32 %arg0) #1 {
  %val = call i32 @llvm.amdgcn.workgroup.id.x()
  store volatile i32 %arg0, i32 addrspace(1)* undef
  call void asm sideeffect "; use $0", "s"(i32 %val)
  ret void
}

; GCN-LABEL: {{^}}other_arg_use_workgroup_id_y:
; GCN: {{flat|global}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, v0
; GCN: ; use s6
define void @other_arg_use_workgroup_id_y(i32 %arg0) #1 {
  %val = call i32 @llvm.amdgcn.workgroup.id.y()
  store volatile i32 %arg0, i32 addrspace(1)* undef
  call void asm sideeffect "; use $0", "s"(i32 %val)
  ret void
}

; GCN-LABEL: {{^}}other_arg_use_workgroup_id_z:
; GCN: {{flat|global}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, v0
; GCN: ; use s6
define void @other_arg_use_workgroup_id_z(i32 %arg0) #1 {
  %val = call i32 @llvm.amdgcn.workgroup.id.z()
  store volatile i32 %arg0, i32 addrspace(1)* undef
  call void asm sideeffect "; use $0", "s"(i32 %val)
  ret void
}

; GCN-LABEL: {{^}}kern_indirect_other_arg_use_workgroup_id_x:
; GCN: enable_sgpr_workgroup_id_x = 1
; GCN: enable_sgpr_workgroup_id_y = 0
; GCN: enable_sgpr_workgroup_id_z = 0

; GCN-DAG: s_mov_b32 s33, s7
; GCN-DAG: v_mov_b32_e32 v0, 0x22b

; GCN-NOT: s6
; GCN: s_mov_b32 s4, s33
; GCN-NOT: s6
; GCN-DAG: s_mov_b32 s32, s33
; GCN: s_swappc_b64
define amdgpu_kernel void @kern_indirect_other_arg_use_workgroup_id_x() #1 {
  call void @other_arg_use_workgroup_id_x(i32 555)
  ret void
}

; GCN-LABEL: {{^}}kern_indirect_other_arg_use_workgroup_id_y:
; GCN: enable_sgpr_workgroup_id_x = 1
; GCN: enable_sgpr_workgroup_id_y = 1
; GCN: enable_sgpr_workgroup_id_z = 0

; GCN-DAG: s_mov_b32 s33, s8
; GCN-DAG: v_mov_b32_e32 v0, 0x22b
; GCN: s_mov_b32 s4, s33
; GCN-DAG: s_mov_b32 s6, s7
; GCN-DAG: s_mov_b32 s32, s33
; GCN: s_swappc_b64
define amdgpu_kernel void @kern_indirect_other_arg_use_workgroup_id_y() #1 {
  call void @other_arg_use_workgroup_id_y(i32 555)
  ret void
}

; GCN-LABEL: {{^}}kern_indirect_other_arg_use_workgroup_id_z:
; GCN: enable_sgpr_workgroup_id_x = 1
; GCN: enable_sgpr_workgroup_id_y = 0
; GCN: enable_sgpr_workgroup_id_z = 1

; GCN: s_mov_b32 s33, s8
; GCN-DAG: v_mov_b32_e32 v0, 0x22b
; GCN: s_mov_b32 s4, s33
; GCN-DAG: s_mov_b32 s6, s7

; GCN: s_mov_b32 s32, s33
; GCN: s_swappc_b64
define amdgpu_kernel void @kern_indirect_other_arg_use_workgroup_id_z() #1 {
  call void @other_arg_use_workgroup_id_z(i32 555)
  ret void
}

; GCN-LABEL: {{^}}use_every_sgpr_input:
; GCN: buffer_store_dword v{{[0-9]+}}, off, s[0:3], s5 offset:4
; GCN: s_load_dword s{{[0-9]+}}, s[6:7], 0x0
; GCN: s_load_dword s{{[0-9]+}}, s[8:9], 0x0
; GCN: s_load_dword s{{[0-9]+}}, s[10:11], 0x0
; GCN: ; use s[12:13]
; GCN: ; use s14
; GCN: ; use s15
; GCN: ; use s16
define void @use_every_sgpr_input() #1 {
  %alloca = alloca i32, align 4
  store volatile i32 0, i32* %alloca

  %dispatch_ptr = call noalias i8 addrspace(2)* @llvm.amdgcn.dispatch.ptr() #0
  %dispatch_ptr.bc = bitcast i8 addrspace(2)* %dispatch_ptr to i32 addrspace(2)*
  %val0 = load volatile i32, i32 addrspace(2)* %dispatch_ptr.bc

  %queue_ptr = call noalias i8 addrspace(2)* @llvm.amdgcn.queue.ptr() #0
  %queue_ptr.bc = bitcast i8 addrspace(2)* %queue_ptr to i32 addrspace(2)*
  %val1 = load volatile i32, i32 addrspace(2)* %queue_ptr.bc

  %kernarg_segment_ptr = call noalias i8 addrspace(2)* @llvm.amdgcn.kernarg.segment.ptr() #0
  %kernarg_segment_ptr.bc = bitcast i8 addrspace(2)* %kernarg_segment_ptr to i32 addrspace(2)*
  %val2 = load volatile i32, i32 addrspace(2)* %kernarg_segment_ptr.bc

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

; GCN: s_mov_b32 s33, s17
; GCN: s_mov_b64 s[12:13], s[10:11]
; GCN: s_mov_b64 s[10:11], s[8:9]
; GCN: s_mov_b64 s[8:9], s[6:7]
; GCN: s_mov_b64 s[6:7], s[4:5]
; GCN: s_mov_b32 s4, s33
; GCN: s_mov_b32 s32, s33
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
define void @func_indirect_use_every_sgpr_input() #1 {
  call void @use_every_sgpr_input()
  ret void
}

; GCN-LABEL: {{^}}func_use_every_sgpr_input_call_use_workgroup_id_xyz:
; GCN-DAG: s_mov_b32 s6, s14
; GCN-DAG: s_mov_b32 s7, s15
; GCN-DAG: s_mov_b32 s8, s16
; GCN: s_swappc_b64
define void @func_use_every_sgpr_input_call_use_workgroup_id_xyz() #1 {
  %alloca = alloca i32, align 4
  store volatile i32 0, i32* %alloca

  %dispatch_ptr = call noalias i8 addrspace(2)* @llvm.amdgcn.dispatch.ptr() #0
  %dispatch_ptr.bc = bitcast i8 addrspace(2)* %dispatch_ptr to i32 addrspace(2)*
  %val0 = load volatile i32, i32 addrspace(2)* %dispatch_ptr.bc

  %queue_ptr = call noalias i8 addrspace(2)* @llvm.amdgcn.queue.ptr() #0
  %queue_ptr.bc = bitcast i8 addrspace(2)* %queue_ptr to i32 addrspace(2)*
  %val1 = load volatile i32, i32 addrspace(2)* %queue_ptr.bc

  %kernarg_segment_ptr = call noalias i8 addrspace(2)* @llvm.amdgcn.kernarg.segment.ptr() #0
  %kernarg_segment_ptr.bc = bitcast i8 addrspace(2)* %kernarg_segment_ptr to i32 addrspace(2)*
  %val2 = load volatile i32, i32 addrspace(2)* %kernarg_segment_ptr.bc

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
; GCN: s_mov_b32 s5, s32
; GCN: s_add_u32 s32, s32, 0x300

; GCN-DAG: s_mov_b32 [[SAVE_X:s[0-9]+]], s14
; GCN-DAG: s_mov_b32 [[SAVE_Y:s[0-9]+]], s15
; GCN-DAG: s_mov_b32 [[SAVE_Z:s[0-9]+]], s16
; GCN-DAG: s_mov_b64 {{s\[[0-9]+:[0-9]+\]}}, s[6:7]
; GCN-DAG: s_mov_b64 {{s\[[0-9]+:[0-9]+\]}}, s[8:9]
; GCN-DAG: s_mov_b64 {{s\[[0-9]+:[0-9]+\]}}, s[10:11]

; GCN-DAG: s_mov_b32 s6, [[SAVE_X]]
; GCN-DAG: s_mov_b32 s7, [[SAVE_Y]]
; GCN-DAG: s_mov_b32 s8, [[SAVE_Z]]
; GCN: s_swappc_b64

; GCN: buffer_store_dword v{{[0-9]+}}, off, s[0:3], s5 offset:4
; GCN: s_load_dword s{{[0-9]+}},
; GCN: s_load_dword s{{[0-9]+}},
; GCN: s_load_dword s{{[0-9]+}},
; GCN: ; use
; GCN: ; use [[SAVE_X]]
; GCN: ; use [[SAVE_Y]]
; GCN: ; use [[SAVE_Z]]
define void @func_use_every_sgpr_input_call_use_workgroup_id_xyz_spill() #1 {
  %alloca = alloca i32, align 4
  call void @use_workgroup_id_xyz()

  store volatile i32 0, i32* %alloca

  %dispatch_ptr = call noalias i8 addrspace(2)* @llvm.amdgcn.dispatch.ptr() #0
  %dispatch_ptr.bc = bitcast i8 addrspace(2)* %dispatch_ptr to i32 addrspace(2)*
  %val0 = load volatile i32, i32 addrspace(2)* %dispatch_ptr.bc

  %queue_ptr = call noalias i8 addrspace(2)* @llvm.amdgcn.queue.ptr() #0
  %queue_ptr.bc = bitcast i8 addrspace(2)* %queue_ptr to i32 addrspace(2)*
  %val1 = load volatile i32, i32 addrspace(2)* %queue_ptr.bc

  %kernarg_segment_ptr = call noalias i8 addrspace(2)* @llvm.amdgcn.kernarg.segment.ptr() #0
  %kernarg_segment_ptr.bc = bitcast i8 addrspace(2)* %kernarg_segment_ptr to i32 addrspace(2)*
  %val2 = load volatile i32, i32 addrspace(2)* %kernarg_segment_ptr.bc

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
declare noalias i8 addrspace(2)* @llvm.amdgcn.queue.ptr() #0
declare noalias i8 addrspace(2)* @llvm.amdgcn.kernarg.segment.ptr() #0
declare i64 @llvm.amdgcn.dispatch.id() #0
declare noalias i8 addrspace(2)* @llvm.amdgcn.dispatch.ptr() #0

attributes #0 = { nounwind readnone speculatable }
attributes #1 = { nounwind noinline }
