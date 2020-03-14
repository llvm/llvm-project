; RUN: llc -amdgpu-fixed-function-abi -mtriple=amdgcn-amd-amdhsa -mcpu=kaveri -enable-ipra=0 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,CIVI %s
; RUN: llc -amdgpu-fixed-function-abi -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -enable-ipra=0 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9 %s

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

; GCN-LABEL: {{^}}use_queue_ptr:
; GCN: v_mov_b32_e32 v[[LO:[0-9]+]], s6
; GCN: v_mov_b32_e32 v[[HI:[0-9]+]], s7
; GCN: {{flat|global}}_load_dword v{{[0-9]+}}, v{{\[}}[[LO]]:[[HI]]{{\]}}
define hidden void @use_queue_ptr() #1 {
  %queue_ptr = call noalias i8 addrspace(4)* @llvm.amdgcn.queue.ptr() #0
  %header_ptr = bitcast i8 addrspace(4)* %queue_ptr to i32 addrspace(4)*
  %value = load volatile i32, i32 addrspace(4)* %header_ptr
  ret void
}

; GCN-LABEL: {{^}}use_kernarg_segment_ptr:
; GCN: s_mov_b64 [[PTR:s\[[0-9]+:[0-9]+\]]], 0
; GCN: s_load_dword s{{[0-9]+}}, [[PTR]], 0x0
define hidden void @use_kernarg_segment_ptr() #1 {
  %kernarg_segment_ptr = call noalias i8 addrspace(4)* @llvm.amdgcn.kernarg.segment.ptr() #0
  %header_ptr = bitcast i8 addrspace(4)* %kernarg_segment_ptr to i32 addrspace(4)*
  %value = load volatile i32, i32 addrspace(4)* %header_ptr
  ret void
}

; GCN-LABEL: {{^}}use_implicitarg_ptr:
; GCN: v_mov_b32_e32 v[[LO:[0-9]+]], s8
; GCN: v_mov_b32_e32 v[[HI:[0-9]+]], s9
; GCN: {{flat|global}}_load_dword v{{[0-9]+}}, v{{\[}}[[LO]]:[[HI]]{{\]}}
define hidden void @use_implicitarg_ptr() #1 {
  %implicit.arg.ptr = call noalias i8 addrspace(4)* @llvm.amdgcn.implicitarg.ptr() #0
  %header_ptr = bitcast i8 addrspace(4)* %implicit.arg.ptr to i32 addrspace(4)*
  %value = load volatile i32, i32 addrspace(4)* %header_ptr
  ret void
}

; GCN-LABEL: {{^}}use_dispatch_id:
; GCN: ; use s[10:11]
define hidden void @use_dispatch_id() #1 {
  %id = call i64 @llvm.amdgcn.dispatch.id()
  call void asm sideeffect "; use $0", "s"(i64 %id)
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
  store volatile i32 0, i32 addrspace(5)* %alloca
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

; Argument is in right place already
; GCN-LABEL: {{^}}func_indirect_use_workgroup_id_x:
; GCN-NOT: s12
; GCN-NOT: s13
; GCN-NOT: s14
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
; GCN: ; use s12
define hidden void @other_arg_use_workgroup_id_x(i32 %arg0) #1 {
  %val = call i32 @llvm.amdgcn.workgroup.id.x()
  store volatile i32 %arg0, i32 addrspace(1)* undef
  call void asm sideeffect "; use $0", "s"(i32 %val)
  ret void
}

; GCN-LABEL: {{^}}other_arg_use_workgroup_id_y:
; GCN: {{flat|global}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, v0
; GCN: ; use s13
define hidden void @other_arg_use_workgroup_id_y(i32 %arg0) #1 {
  %val = call i32 @llvm.amdgcn.workgroup.id.y()
  store volatile i32 %arg0, i32 addrspace(1)* undef
  call void asm sideeffect "; use $0", "s"(i32 %val)
  ret void
}

; GCN-LABEL: {{^}}other_arg_use_workgroup_id_z:
; GCN: {{flat|global}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, v0
; GCN: ; use s14
define hidden void @other_arg_use_workgroup_id_z(i32 %arg0) #1 {
  %val = call i32 @llvm.amdgcn.workgroup.id.z()
  store volatile i32 %arg0, i32 addrspace(1)* undef
  call void asm sideeffect "; use $0", "s"(i32 %val)
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
; GCN: s_mov_b32 s33, s17
; GCN: s_mov_b32 s12, s14
; GCN: s_mov_b32 s13, s15
; GCN: s_mov_b32 s14, s16
; GCN: s_mov_b32 s32, s33
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
; GCN: .amdhsa_system_vgpr_workitem_id 2
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
  store volatile i32 0, i32 addrspace(5)* %alloca

  %dispatch_ptr = call noalias i8 addrspace(4)* @llvm.amdgcn.dispatch.ptr() #0
  %dispatch_ptr.bc = bitcast i8 addrspace(4)* %dispatch_ptr to i32 addrspace(4)*
  %val0 = load volatile i32, i32 addrspace(4)* %dispatch_ptr.bc

  %queue_ptr = call noalias i8 addrspace(4)* @llvm.amdgcn.queue.ptr() #0
  %queue_ptr.bc = bitcast i8 addrspace(4)* %queue_ptr to i32 addrspace(4)*
  %val1 = load volatile i32, i32 addrspace(4)* %queue_ptr.bc

  %kernarg_segment_ptr = call noalias i8 addrspace(4)* @llvm.amdgcn.implicitarg.ptr() #0
  %kernarg_segment_ptr.bc = bitcast i8 addrspace(4)* %kernarg_segment_ptr to i32 addrspace(4)*
  %val2 = load volatile i32, i32 addrspace(4)* %kernarg_segment_ptr.bc

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
declare noalias i8 addrspace(4)* @llvm.amdgcn.queue.ptr() #0
declare noalias i8 addrspace(4)* @llvm.amdgcn.kernarg.segment.ptr() #0
declare noalias i8 addrspace(4)* @llvm.amdgcn.implicitarg.ptr() #0
declare i64 @llvm.amdgcn.dispatch.id() #0
declare noalias i8 addrspace(4)* @llvm.amdgcn.dispatch.ptr() #0

attributes #0 = { nounwind readnone speculatable }
attributes #1 = { nounwind noinline }
