; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=kaveri -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN %s

; GCN-LABEL: {{^}}use_workitem_id_x:
; GCN: s_waitcnt
; GCN-NEXT: flat_store_dword v{{\[[0-9]:[0-9]+\]}}, v0
; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @use_workitem_id_x() #1 {
  %val = call i32 @llvm.amdgcn.workitem.id.x()
  store volatile i32 %val, i32 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}use_workitem_id_y:
; GCN: s_waitcnt
; GCN-NEXT: flat_store_dword v{{\[[0-9]:[0-9]+\]}}, v0
; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @use_workitem_id_y() #1 {
  %val = call i32 @llvm.amdgcn.workitem.id.y()
  store volatile i32 %val, i32 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}use_workitem_id_z:
; GCN: s_waitcnt
; GCN-NEXT: flat_store_dword v{{\[[0-9]:[0-9]+\]}}, v0
; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @use_workitem_id_z() #1 {
  %val = call i32 @llvm.amdgcn.workitem.id.z()
  store volatile i32 %val, i32 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}use_workitem_id_xy:
; GCN: s_waitcnt
; GCN-NEXT: flat_store_dword v{{\[[0-9]:[0-9]+\]}}, v0
; GCN-NEXT: flat_store_dword v{{\[[0-9]:[0-9]+\]}}, v1
; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @use_workitem_id_xy() #1 {
  %val0 = call i32 @llvm.amdgcn.workitem.id.x()
  %val1 = call i32 @llvm.amdgcn.workitem.id.y()
  store volatile i32 %val0, i32 addrspace(1)* undef
  store volatile i32 %val1, i32 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}use_workitem_id_xyz:
; GCN: s_waitcnt
; GCN-NEXT: flat_store_dword v{{\[[0-9]:[0-9]+\]}}, v0
; GCN-NEXT: flat_store_dword v{{\[[0-9]:[0-9]+\]}}, v1
; GCN-NEXT: flat_store_dword v{{\[[0-9]:[0-9]+\]}}, v2
; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @use_workitem_id_xyz() #1 {
  %val0 = call i32 @llvm.amdgcn.workitem.id.x()
  %val1 = call i32 @llvm.amdgcn.workitem.id.y()
  %val2 = call i32 @llvm.amdgcn.workitem.id.z()
  store volatile i32 %val0, i32 addrspace(1)* undef
  store volatile i32 %val1, i32 addrspace(1)* undef
  store volatile i32 %val2, i32 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}use_workitem_id_xz:
; GCN: s_waitcnt
; GCN-NEXT: flat_store_dword v{{\[[0-9]:[0-9]+\]}}, v0
; GCN-NEXT: flat_store_dword v{{\[[0-9]:[0-9]+\]}}, v1
; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @use_workitem_id_xz() #1 {
  %val0 = call i32 @llvm.amdgcn.workitem.id.x()
  %val1 = call i32 @llvm.amdgcn.workitem.id.z()
  store volatile i32 %val0, i32 addrspace(1)* undef
  store volatile i32 %val1, i32 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}use_workitem_id_yz:
; GCN: s_waitcnt
; GCN-NEXT: flat_store_dword v{{\[[0-9]:[0-9]+\]}}, v0
; GCN-NEXT: flat_store_dword v{{\[[0-9]:[0-9]+\]}}, v1
; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @use_workitem_id_yz() #1 {
  %val0 = call i32 @llvm.amdgcn.workitem.id.y()
  %val1 = call i32 @llvm.amdgcn.workitem.id.z()
  store volatile i32 %val0, i32 addrspace(1)* undef
  store volatile i32 %val1, i32 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}kern_indirect_use_workitem_id_x:
; GCN: enable_vgpr_workitem_id = 0

; GCN-NOT: v0
; GCN: s_swappc_b64
; GCN-NOT: v0
define amdgpu_kernel void @kern_indirect_use_workitem_id_x() #1 {
  call void @use_workitem_id_x()
  ret void
}

; GCN-LABEL: {{^}}kern_indirect_use_workitem_id_y:
; GCN: enable_vgpr_workitem_id = 1

; GCN-NOT: v0
; GCN-NOT: v1
; GCN: v_mov_b32_e32 v0, v1
; GCN-NOT: v0
; GCN-NOT: v1
; GCN: s_swappc_b64
define amdgpu_kernel void @kern_indirect_use_workitem_id_y() #1 {
  call void @use_workitem_id_y()
  ret void
}

; GCN-LABEL: {{^}}kern_indirect_use_workitem_id_z:
; GCN: enable_vgpr_workitem_id = 2

; GCN-NOT: v0
; GCN-NOT: v2
; GCN: v_mov_b32_e32 v0, v2
; GCN-NOT: v0
; GCN-NOT: v2
; GCN: s_swappc_b64
define amdgpu_kernel void @kern_indirect_use_workitem_id_z() #1 {
  call void @use_workitem_id_z()
  ret void
}

; GCN-LABEL: {{^}}func_indirect_use_workitem_id_x:
; GCN-NOT: v0
; GCN: s_swappc_b64
; GCN-NOT: v0
define void @func_indirect_use_workitem_id_x() #1 {
  call void @use_workitem_id_x()
  ret void
}

; GCN-LABEL: {{^}}func_indirect_use_workitem_id_y:
; GCN-NOT: v0
; GCN: s_swappc_b64
; GCN-NOT: v0
define void @func_indirect_use_workitem_id_y() #1 {
  call void @use_workitem_id_y()
  ret void
}

; GCN-LABEL: {{^}}func_indirect_use_workitem_id_z:
; GCN-NOT: v0
; GCN: s_swappc_b64
; GCN-NOT: v0
define void @func_indirect_use_workitem_id_z() #1 {
  call void @use_workitem_id_z()
  ret void
}

; GCN-LABEL: {{^}}other_arg_use_workitem_id_x:
; GCN: s_waitcnt
; GCN-NEXT: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, v0
; GCN-NEXT: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, v1
define void @other_arg_use_workitem_id_x(i32 %arg0) #1 {
  %val = call i32 @llvm.amdgcn.workitem.id.x()
  store volatile i32 %arg0, i32 addrspace(1)* undef
  store volatile i32 %val, i32 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}other_arg_use_workitem_id_y:
; GCN: s_waitcnt
; GCN-NEXT: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, v0
; GCN-NEXT: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, v1
define void @other_arg_use_workitem_id_y(i32 %arg0) #1 {
  %val = call i32 @llvm.amdgcn.workitem.id.y()
  store volatile i32 %arg0, i32 addrspace(1)* undef
  store volatile i32 %val, i32 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}other_arg_use_workitem_id_z:
; GCN: s_waitcnt
; GCN-NEXT: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, v0
; GCN-NEXT: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, v1
define void @other_arg_use_workitem_id_z(i32 %arg0) #1 {
  %val = call i32 @llvm.amdgcn.workitem.id.z()
  store volatile i32 %arg0, i32 addrspace(1)* undef
  store volatile i32 %val, i32 addrspace(1)* undef
  ret void
}


; GCN-LABEL: {{^}}kern_indirect_other_arg_use_workitem_id_x:
; GCN: enable_vgpr_workitem_id = 0

; GCN: v_mov_b32_e32 v1, v0
; GCN: v_mov_b32_e32 v0, 0x22b
; GCN: s_swappc_b64
define amdgpu_kernel void @kern_indirect_other_arg_use_workitem_id_x() #1 {
  call void @other_arg_use_workitem_id_x(i32 555)
  ret void
}


; GCN-LABEL: {{^}}kern_indirect_other_arg_use_workitem_id_y:
; GCN: enable_vgpr_workitem_id = 1

; GCN-NOT: v1
; GCN: v_mov_b32_e32 v0, 0x22b
; GCN-NOT: v1
; GCN: s_swappc_b64
; GCN-NOT: v0
define amdgpu_kernel void @kern_indirect_other_arg_use_workitem_id_y() #1 {
  call void @other_arg_use_workitem_id_y(i32 555)
  ret void
}

; GCN-LABEL: {{^}}kern_indirect_other_arg_use_workitem_id_z:
; GCN: enable_vgpr_workitem_id = 2

; GCN: v_mov_b32_e32 v0, 0x22b
; GCN: v_mov_b32_e32 v1, v2
; GCN: s_swappc_b64
; GCN-NOT: v0
define amdgpu_kernel void @kern_indirect_other_arg_use_workitem_id_z() #1 {
  call void @other_arg_use_workitem_id_z(i32 555)
  ret void
}

; GCN-LABEL: {{^}}too_many_args_use_workitem_id_x:
; GCN: s_mov_b32 s5, s32
; GCN: buffer_store_dword v32, off, s[0:3], s5 offset:8 ; 4-byte Folded Spill
; GCN: buffer_load_dword v32, off, s[0:3], s5 offset:4{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+]}}, v32

; GCN: buffer_load_dword v32, off, s[0:3], s5 offset:8 ; 4-byte Folded Reload
; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @too_many_args_use_workitem_id_x(
  i32 %arg0, i32 %arg1, i32 %arg2, i32 %arg3, i32 %arg4, i32 %arg5, i32 %arg6, i32 %arg7,
  i32 %arg8, i32 %arg9, i32 %arg10, i32 %arg11, i32 %arg12, i32 %arg13, i32 %arg14, i32 %arg15,
  i32 %arg16, i32 %arg17, i32 %arg18, i32 %arg19, i32 %arg20, i32 %arg21, i32 %arg22, i32 %arg23,
  i32 %arg24, i32 %arg25, i32 %arg26, i32 %arg27, i32 %arg28, i32 %arg29, i32 %arg30, i32 %arg31) #1 {
  %val = call i32 @llvm.amdgcn.workitem.id.x()
  store volatile i32 %val, i32 addrspace(1)* undef

  store volatile i32 %arg0, i32 addrspace(1)* undef
  store volatile i32 %arg1, i32 addrspace(1)* undef
  store volatile i32 %arg2, i32 addrspace(1)* undef
  store volatile i32 %arg3, i32 addrspace(1)* undef
  store volatile i32 %arg4, i32 addrspace(1)* undef
  store volatile i32 %arg5, i32 addrspace(1)* undef
  store volatile i32 %arg6, i32 addrspace(1)* undef
  store volatile i32 %arg7, i32 addrspace(1)* undef

  store volatile i32 %arg8, i32 addrspace(1)* undef
  store volatile i32 %arg9, i32 addrspace(1)* undef
  store volatile i32 %arg10, i32 addrspace(1)* undef
  store volatile i32 %arg11, i32 addrspace(1)* undef
  store volatile i32 %arg12, i32 addrspace(1)* undef
  store volatile i32 %arg13, i32 addrspace(1)* undef
  store volatile i32 %arg14, i32 addrspace(1)* undef
  store volatile i32 %arg15, i32 addrspace(1)* undef

  store volatile i32 %arg16, i32 addrspace(1)* undef
  store volatile i32 %arg17, i32 addrspace(1)* undef
  store volatile i32 %arg18, i32 addrspace(1)* undef
  store volatile i32 %arg19, i32 addrspace(1)* undef
  store volatile i32 %arg20, i32 addrspace(1)* undef
  store volatile i32 %arg21, i32 addrspace(1)* undef
  store volatile i32 %arg22, i32 addrspace(1)* undef
  store volatile i32 %arg23, i32 addrspace(1)* undef

  store volatile i32 %arg24, i32 addrspace(1)* undef
  store volatile i32 %arg25, i32 addrspace(1)* undef
  store volatile i32 %arg26, i32 addrspace(1)* undef
  store volatile i32 %arg27, i32 addrspace(1)* undef
  store volatile i32 %arg28, i32 addrspace(1)* undef
  store volatile i32 %arg29, i32 addrspace(1)* undef
  store volatile i32 %arg30, i32 addrspace(1)* undef
  store volatile i32 %arg31, i32 addrspace(1)* undef

  ret void
}

; GCN-LABEL: {{^}}kern_call_too_many_args_use_workitem_id_x:
; GCN: enable_vgpr_workitem_id = 0

; GCN: s_mov_b32 s33, s7
; GCN: s_mov_b32 s32, s33
; GCN: buffer_store_dword v0, off, s[0:3], s32 offset:8
; GCN: s_mov_b32 s4, s33
; GCN: s_swappc_b64
define amdgpu_kernel void @kern_call_too_many_args_use_workitem_id_x() #1 {
  call void @too_many_args_use_workitem_id_x(
    i32 10, i32 20, i32 30, i32 40,
    i32 50, i32 60, i32 70, i32 80,
    i32 90, i32 100, i32 110, i32 120,
    i32 130, i32 140, i32 150, i32 160,
    i32 170, i32 180, i32 190, i32 200,
    i32 210, i32 220, i32 230, i32 240,
    i32 250, i32 260, i32 270, i32 280,
    i32 290, i32 300, i32 310, i32 320)
  ret void
}

; GCN-LABEL: {{^}}func_call_too_many_args_use_workitem_id_x:
; GCN: s_mov_b32 s5, s32
; GCN: buffer_store_dword v1, off, s[0:3], s32 offset:8
; GCN: s_swappc_b64
define void @func_call_too_many_args_use_workitem_id_x(i32 %arg0) #1 {
  store volatile i32 %arg0, i32 addrspace(1)* undef
  call void @too_many_args_use_workitem_id_x(
    i32 10, i32 20, i32 30, i32 40,
    i32 50, i32 60, i32 70, i32 80,
    i32 90, i32 100, i32 110, i32 120,
    i32 130, i32 140, i32 150, i32 160,
    i32 170, i32 180, i32 190, i32 200,
    i32 210, i32 220, i32 230, i32 240,
    i32 250, i32 260, i32 270, i32 280,
    i32 290, i32 300, i32 310, i32 320)
  ret void
}

; Requires loading and storing to stack slot.
; GCN-LABEL: {{^}}too_many_args_call_too_many_args_use_workitem_id_x:
; GCN: buffer_store_dword v32, off, s[0:3], s5 offset:8 ; 4-byte Folded Spill
; GCN: buffer_load_dword v32, off, s[0:3], s5 offset:4
; GCN: s_add_u32 s32, s32, 0x400{{$}}

; GCN: buffer_store_dword v32, off, s[0:3], s32 offset:8{{$}}

; GCN: s_swappc_b64

; GCN: buffer_load_dword v32, off, s[0:3], s5 offset:8 ; 4-byte Folded Reload
; GCN: s_sub_u32 s32, s32, 0x400{{$}}
; GCN: s_setpc_b64
define void @too_many_args_call_too_many_args_use_workitem_id_x(
  i32 %arg0, i32 %arg1, i32 %arg2, i32 %arg3, i32 %arg4, i32 %arg5, i32 %arg6, i32 %arg7,
  i32 %arg8, i32 %arg9, i32 %arg10, i32 %arg11, i32 %arg12, i32 %arg13, i32 %arg14, i32 %arg15,
  i32 %arg16, i32 %arg17, i32 %arg18, i32 %arg19, i32 %arg20, i32 %arg21, i32 %arg22, i32 %arg23,
  i32 %arg24, i32 %arg25, i32 %arg26, i32 %arg27, i32 %arg28, i32 %arg29, i32 %arg30, i32 %arg31) #1 {
  call void @too_many_args_use_workitem_id_x(
    i32 %arg0, i32 %arg1, i32 %arg2, i32 %arg3, i32 %arg4, i32 %arg5, i32 %arg6, i32 %arg7,
    i32 %arg8, i32 %arg9, i32 %arg10, i32 %arg11, i32 %arg12, i32 %arg13, i32 %arg14, i32 %arg15,
    i32 %arg16, i32 %arg17, i32 %arg18, i32 %arg19, i32 %arg20, i32 %arg21, i32 %arg22, i32 %arg23,
    i32 %arg24, i32 %arg25, i32 %arg26, i32 %arg27, i32 %arg28, i32 %arg29, i32 %arg30, i32 %arg31)
  ret void
}

; stack layout:
; frame[0] = emergency stack slot
; frame[1] = byval arg32
; frame[2] = stack passed workitem ID x
; frame[3] = VGPR spill slot

; GCN-LABEL: {{^}}too_many_args_use_workitem_id_x_byval:
; GCN: buffer_store_dword v32, off, s[0:3], s5 offset:12 ; 4-byte Folded Spill
; GCN: buffer_load_dword v32, off, s[0:3], s5 offset:8
; GCN-NEXT: s_waitcnt
; GCN-NEXT: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, v32
; GCN: buffer_load_dword v0, off, s[0:3], s5 offset:4
; GCN: buffer_load_dword v32, off, s[0:3], s5 offset:12 ; 4-byte Folded Reload
; GCN: s_setpc_b64
define void @too_many_args_use_workitem_id_x_byval(
  i32 %arg0, i32 %arg1, i32 %arg2, i32 %arg3, i32 %arg4, i32 %arg5, i32 %arg6, i32 %arg7,
  i32 %arg8, i32 %arg9, i32 %arg10, i32 %arg11, i32 %arg12, i32 %arg13, i32 %arg14, i32 %arg15,
  i32 %arg16, i32 %arg17, i32 %arg18, i32 %arg19, i32 %arg20, i32 %arg21, i32 %arg22, i32 %arg23,
  i32 %arg24, i32 %arg25, i32 %arg26, i32 %arg27, i32 %arg28, i32 %arg29, i32 %arg30, i32 %arg31, i32* byval %arg32) #1 {
  %val = call i32 @llvm.amdgcn.workitem.id.x()
  store volatile i32 %val, i32 addrspace(1)* undef

  store volatile i32 %arg0, i32 addrspace(1)* undef
  store volatile i32 %arg1, i32 addrspace(1)* undef
  store volatile i32 %arg2, i32 addrspace(1)* undef
  store volatile i32 %arg3, i32 addrspace(1)* undef
  store volatile i32 %arg4, i32 addrspace(1)* undef
  store volatile i32 %arg5, i32 addrspace(1)* undef
  store volatile i32 %arg6, i32 addrspace(1)* undef
  store volatile i32 %arg7, i32 addrspace(1)* undef

  store volatile i32 %arg8, i32 addrspace(1)* undef
  store volatile i32 %arg9, i32 addrspace(1)* undef
  store volatile i32 %arg10, i32 addrspace(1)* undef
  store volatile i32 %arg11, i32 addrspace(1)* undef
  store volatile i32 %arg12, i32 addrspace(1)* undef
  store volatile i32 %arg13, i32 addrspace(1)* undef
  store volatile i32 %arg14, i32 addrspace(1)* undef
  store volatile i32 %arg15, i32 addrspace(1)* undef

  store volatile i32 %arg16, i32 addrspace(1)* undef
  store volatile i32 %arg17, i32 addrspace(1)* undef
  store volatile i32 %arg18, i32 addrspace(1)* undef
  store volatile i32 %arg19, i32 addrspace(1)* undef
  store volatile i32 %arg20, i32 addrspace(1)* undef
  store volatile i32 %arg21, i32 addrspace(1)* undef
  store volatile i32 %arg22, i32 addrspace(1)* undef
  store volatile i32 %arg23, i32 addrspace(1)* undef

  store volatile i32 %arg24, i32 addrspace(1)* undef
  store volatile i32 %arg25, i32 addrspace(1)* undef
  store volatile i32 %arg26, i32 addrspace(1)* undef
  store volatile i32 %arg27, i32 addrspace(1)* undef
  store volatile i32 %arg28, i32 addrspace(1)* undef
  store volatile i32 %arg29, i32 addrspace(1)* undef
  store volatile i32 %arg30, i32 addrspace(1)* undef
  store volatile i32 %arg31, i32 addrspace(1)* undef
  %private = load volatile i32, i32* %arg32
  ret void
}

; frame[0] = emergency stack slot
; frame[1] =

; sp[0] = callee emergency stack slot reservation
; sp[1] = byval
; sp[2] = ??
; sp[3] = stack passed workitem ID x

; GCN-LABEL: {{^}}kern_call_too_many_args_use_workitem_id_x_byval:
; GCN: enable_vgpr_workitem_id = 0

; GCN: s_mov_b32 s33, s7
; GCN: s_add_u32 s32, s33, 0x200{{$}}

; GCN-DAG: s_add_u32 s32, s32, 0x100{{$}}
; GCN-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 0x3e7{{$}}
; GCN: buffer_store_dword [[K]], off, s[0:3], s33 offset:4
; GCN: buffer_store_dword v0, off, s[0:3], s32 offset:12

; GCN: buffer_load_dword [[RELOAD_BYVAL:v[0-9]+]], off, s[0:3], s33 offset:4
; GCN: buffer_store_dword [[RELOAD_BYVAL]], off, s[0:3], s32 offset:4{{$}}
; GCN: v_mov_b32_e32 [[RELOAD_BYVAL]],
; GCN: s_swappc_b64
define amdgpu_kernel void @kern_call_too_many_args_use_workitem_id_x_byval() #1 {
  %alloca = alloca i32, align 4
  store volatile i32 999, i32* %alloca
  call void @too_many_args_use_workitem_id_x_byval(
    i32 10, i32 20, i32 30, i32 40,
    i32 50, i32 60, i32 70, i32 80,
    i32 90, i32 100, i32 110, i32 120,
    i32 130, i32 140, i32 150, i32 160,
    i32 170, i32 180, i32 190, i32 200,
    i32 210, i32 220, i32 230, i32 240,
    i32 250, i32 260, i32 270, i32 280,
    i32 290, i32 300, i32 310, i32 320,
    i32* %alloca)
  ret void
}

; GCN-LABEL: {{^}}func_call_too_many_args_use_workitem_id_x_byval:
; GCN: v_mov_b32_e32 [[K:v[0-9]+]], 0x3e7{{$}}
; GCN: buffer_store_dword [[K]], off, s[0:3], s5 offset:4
; GCN: buffer_store_dword v0, off, s[0:3], s32 offset:12

; GCN: buffer_load_dword [[RELOAD_BYVAL:v[0-9]+]], off, s[0:3], s5 offset:4
; GCN: buffer_store_dword [[RELOAD_BYVAL]], off, s[0:3], s32 offset:4{{$}}
; GCN: v_mov_b32_e32 [[RELOAD_BYVAL]],
; GCN: s_swappc_b64
define void @func_call_too_many_args_use_workitem_id_x_byval() #1 {
  %alloca = alloca i32, align 4
  store volatile i32 999, i32* %alloca
  call void @too_many_args_use_workitem_id_x_byval(
    i32 10, i32 20, i32 30, i32 40,
    i32 50, i32 60, i32 70, i32 80,
    i32 90, i32 100, i32 110, i32 120,
    i32 130, i32 140, i32 150, i32 160,
    i32 170, i32 180, i32 190, i32 200,
    i32 210, i32 220, i32 230, i32 240,
    i32 250, i32 260, i32 270, i32 280,
    i32 290, i32 300, i32 310, i32 320,
    i32* %alloca)
  ret void
}

; GCN-LABEL: {{^}}too_many_args_use_workitem_id_xyz:
; GCN: s_mov_b32 s5, s32
; GCN: buffer_store_dword v32, off, s[0:3], s5 offset:16 ; 4-byte Folded Spill
; GCN: buffer_load_dword v32, off, s[0:3], s5 offset:4{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+]}}, v32
; GCN: buffer_load_dword v32, off, s[0:3], s5 offset:8{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+]}}, v32
; GCN: buffer_load_dword v32, off, s[0:3], s5 offset:12{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+]}}, v32

; GCN: buffer_load_dword v32, off, s[0:3], s5 offset:16 ; 4-byte Folded Reload
; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @too_many_args_use_workitem_id_xyz(
  i32 %arg0, i32 %arg1, i32 %arg2, i32 %arg3, i32 %arg4, i32 %arg5, i32 %arg6, i32 %arg7,
  i32 %arg8, i32 %arg9, i32 %arg10, i32 %arg11, i32 %arg12, i32 %arg13, i32 %arg14, i32 %arg15,
  i32 %arg16, i32 %arg17, i32 %arg18, i32 %arg19, i32 %arg20, i32 %arg21, i32 %arg22, i32 %arg23,
  i32 %arg24, i32 %arg25, i32 %arg26, i32 %arg27, i32 %arg28, i32 %arg29, i32 %arg30, i32 %arg31) #1 {
  %val0 = call i32 @llvm.amdgcn.workitem.id.x()
  store volatile i32 %val0, i32 addrspace(1)* undef
  %val1 = call i32 @llvm.amdgcn.workitem.id.y()
  store volatile i32 %val1, i32 addrspace(1)* undef
  %val2 = call i32 @llvm.amdgcn.workitem.id.z()
  store volatile i32 %val2, i32 addrspace(1)* undef

  store volatile i32 %arg0, i32 addrspace(1)* undef
  store volatile i32 %arg1, i32 addrspace(1)* undef
  store volatile i32 %arg2, i32 addrspace(1)* undef
  store volatile i32 %arg3, i32 addrspace(1)* undef
  store volatile i32 %arg4, i32 addrspace(1)* undef
  store volatile i32 %arg5, i32 addrspace(1)* undef
  store volatile i32 %arg6, i32 addrspace(1)* undef
  store volatile i32 %arg7, i32 addrspace(1)* undef

  store volatile i32 %arg8, i32 addrspace(1)* undef
  store volatile i32 %arg9, i32 addrspace(1)* undef
  store volatile i32 %arg10, i32 addrspace(1)* undef
  store volatile i32 %arg11, i32 addrspace(1)* undef
  store volatile i32 %arg12, i32 addrspace(1)* undef
  store volatile i32 %arg13, i32 addrspace(1)* undef
  store volatile i32 %arg14, i32 addrspace(1)* undef
  store volatile i32 %arg15, i32 addrspace(1)* undef

  store volatile i32 %arg16, i32 addrspace(1)* undef
  store volatile i32 %arg17, i32 addrspace(1)* undef
  store volatile i32 %arg18, i32 addrspace(1)* undef
  store volatile i32 %arg19, i32 addrspace(1)* undef
  store volatile i32 %arg20, i32 addrspace(1)* undef
  store volatile i32 %arg21, i32 addrspace(1)* undef
  store volatile i32 %arg22, i32 addrspace(1)* undef
  store volatile i32 %arg23, i32 addrspace(1)* undef

  store volatile i32 %arg24, i32 addrspace(1)* undef
  store volatile i32 %arg25, i32 addrspace(1)* undef
  store volatile i32 %arg26, i32 addrspace(1)* undef
  store volatile i32 %arg27, i32 addrspace(1)* undef
  store volatile i32 %arg28, i32 addrspace(1)* undef
  store volatile i32 %arg29, i32 addrspace(1)* undef
  store volatile i32 %arg30, i32 addrspace(1)* undef
  store volatile i32 %arg31, i32 addrspace(1)* undef

  ret void
}

; frame[0] = kernel emergency stack slot
; frame[1] = callee emergency stack slot
; frame[2] = ID X
; frame[3] = ID Y
; frame[4] = ID Z

; GCN-LABEL: {{^}}kern_call_too_many_args_use_workitem_id_xyz:
; GCN: enable_vgpr_workitem_id = 2

; GCN: s_mov_b32 s33, s7
; GCN: s_mov_b32 s32, s33

; GCN-DAG: buffer_store_dword v0, off, s[0:3], s32 offset:8
; GCN-DAG: buffer_store_dword v1, off, s[0:3], s32 offset:12
; GCN-DAG: buffer_store_dword v2, off, s[0:3], s32 offset:16
; GCN: s_swappc_b64
define amdgpu_kernel void @kern_call_too_many_args_use_workitem_id_xyz() #1 {
  call void @too_many_args_use_workitem_id_xyz(
    i32 10, i32 20, i32 30, i32 40,
    i32 50, i32 60, i32 70, i32 80,
    i32 90, i32 100, i32 110, i32 120,
    i32 130, i32 140, i32 150, i32 160,
    i32 170, i32 180, i32 190, i32 200,
    i32 210, i32 220, i32 230, i32 240,
    i32 250, i32 260, i32 270, i32 280,
    i32 290, i32 300, i32 310, i32 320)
  ret void
}

; workitem ID X in register, yz on stack
; v31 = workitem ID X
; frame[0] = emergency slot
; frame[1] = workitem Y
; frame[2] = workitem Z

; GCN-LABEL: {{^}}too_many_args_use_workitem_id_x_stack_yz:
; GCN: s_mov_b32 s5, s32
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+]}}, v31
; GCN: buffer_load_dword v31, off, s[0:3], s5 offset:4{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+]}}, v31
; GCN: buffer_load_dword v31, off, s[0:3], s5 offset:8{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+]}}, v31

; GCN: s_waitcnt
; GCN-NEXT: s_setpc_b64
; GCN: ScratchSize: 12
define void @too_many_args_use_workitem_id_x_stack_yz(
  i32 %arg0, i32 %arg1, i32 %arg2, i32 %arg3, i32 %arg4, i32 %arg5, i32 %arg6, i32 %arg7,
  i32 %arg8, i32 %arg9, i32 %arg10, i32 %arg11, i32 %arg12, i32 %arg13, i32 %arg14, i32 %arg15,
  i32 %arg16, i32 %arg17, i32 %arg18, i32 %arg19, i32 %arg20, i32 %arg21, i32 %arg22, i32 %arg23,
  i32 %arg24, i32 %arg25, i32 %arg26, i32 %arg27, i32 %arg28, i32 %arg29, i32 %arg30) #1 {
  %val0 = call i32 @llvm.amdgcn.workitem.id.x()
  store volatile i32 %val0, i32 addrspace(1)* undef
  %val1 = call i32 @llvm.amdgcn.workitem.id.y()
  store volatile i32 %val1, i32 addrspace(1)* undef
  %val2 = call i32 @llvm.amdgcn.workitem.id.z()
  store volatile i32 %val2, i32 addrspace(1)* undef

  store volatile i32 %arg0, i32 addrspace(1)* undef
  store volatile i32 %arg1, i32 addrspace(1)* undef
  store volatile i32 %arg2, i32 addrspace(1)* undef
  store volatile i32 %arg3, i32 addrspace(1)* undef
  store volatile i32 %arg4, i32 addrspace(1)* undef
  store volatile i32 %arg5, i32 addrspace(1)* undef
  store volatile i32 %arg6, i32 addrspace(1)* undef
  store volatile i32 %arg7, i32 addrspace(1)* undef

  store volatile i32 %arg8, i32 addrspace(1)* undef
  store volatile i32 %arg9, i32 addrspace(1)* undef
  store volatile i32 %arg10, i32 addrspace(1)* undef
  store volatile i32 %arg11, i32 addrspace(1)* undef
  store volatile i32 %arg12, i32 addrspace(1)* undef
  store volatile i32 %arg13, i32 addrspace(1)* undef
  store volatile i32 %arg14, i32 addrspace(1)* undef
  store volatile i32 %arg15, i32 addrspace(1)* undef

  store volatile i32 %arg16, i32 addrspace(1)* undef
  store volatile i32 %arg17, i32 addrspace(1)* undef
  store volatile i32 %arg18, i32 addrspace(1)* undef
  store volatile i32 %arg19, i32 addrspace(1)* undef
  store volatile i32 %arg20, i32 addrspace(1)* undef
  store volatile i32 %arg21, i32 addrspace(1)* undef
  store volatile i32 %arg22, i32 addrspace(1)* undef
  store volatile i32 %arg23, i32 addrspace(1)* undef

  store volatile i32 %arg24, i32 addrspace(1)* undef
  store volatile i32 %arg25, i32 addrspace(1)* undef
  store volatile i32 %arg26, i32 addrspace(1)* undef
  store volatile i32 %arg27, i32 addrspace(1)* undef
  store volatile i32 %arg28, i32 addrspace(1)* undef
  store volatile i32 %arg29, i32 addrspace(1)* undef
  store volatile i32 %arg30, i32 addrspace(1)* undef

  ret void
}

; frame[0] = kernel emergency stack slot
; frame[1] = callee emergency stack slot
; frame[2] = ID Y
; frame[3] = ID Z

; GCN-LABEL: {{^}}kern_call_too_many_args_use_workitem_id_x_stack_yz:
; GCN: enable_vgpr_workitem_id = 2

; GCN: s_mov_b32 s33, s7
; GCN: s_mov_b32 s32, s33

; GCN-DAG: v_mov_b32_e32 v31, v0
; GCN-DAG: buffer_store_dword v1, off, s[0:3], s32 offset:8
; GCN-DAG: buffer_store_dword v2, off, s[0:3], s32 offset:12
; GCN: s_swappc_b64
define amdgpu_kernel void @kern_call_too_many_args_use_workitem_id_x_stack_yz() #1 {
  call void @too_many_args_use_workitem_id_x_stack_yz(
    i32 10, i32 20, i32 30, i32 40,
    i32 50, i32 60, i32 70, i32 80,
    i32 90, i32 100, i32 110, i32 120,
    i32 130, i32 140, i32 150, i32 160,
    i32 170, i32 180, i32 190, i32 200,
    i32 210, i32 220, i32 230, i32 240,
    i32 250, i32 260, i32 270, i32 280,
    i32 290, i32 300, i32 310)
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #0
declare i32 @llvm.amdgcn.workitem.id.y() #0
declare i32 @llvm.amdgcn.workitem.id.z() #0

attributes #0 = { nounwind readnone speculatable }
attributes #1 = { nounwind noinline }
