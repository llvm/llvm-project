; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 < %s | FileCheck %s

; We have an indirect call with a known set of callees, which are
; known to not need any special inputs. The ABI still needs to use the
; register

; FIXME: Passing real values for workitem ID, and 0s that can be undef

; CHECK-LABEL: indirect_call_known_no_special_inputs:

; CHECK-DAG: s_cselect_b64 [[CALL_TARGET:s\[[0-9]+:[0-9]+\]]]
; CHECK-DAG: s_mov_b64 s[8:9], 0
; CHECK-DAG: s_mov_b32 s12, s6
; CHECK-DAG: v_mov_b32_e32 v31, v0
; CHECK: s_swappc_b64 s[30:31], [[CALL_TARGET]]

; CHECK: .amdhsa_kernarg_size 0
; CHECK-NEXT: .amdhsa_user_sgpr_private_segment_buffer 1
; CHECK-NEXT: .amdhsa_user_sgpr_dispatch_ptr 0
; CHECK-NEXT: .amdhsa_user_sgpr_queue_ptr 0
; CHECK-NEXT: .amdhsa_user_sgpr_kernarg_segment_ptr 0
; CHECK-NEXT: .amdhsa_user_sgpr_dispatch_id 0
; CHECK-NEXT: .amdhsa_user_sgpr_flat_scratch_init 1
; CHECK-NEXT: .amdhsa_user_sgpr_private_segment_size 0
; CHECK-NEXT: .amdhsa_system_sgpr_private_segment_wavefront_offset 1
; CHECK-NEXT: .amdhsa_system_sgpr_workgroup_id_x 1
; CHECK-NEXT: .amdhsa_system_sgpr_workgroup_id_y 0
; CHECK-NEXT: .amdhsa_system_sgpr_workgroup_id_z 0
; CHECK-NEXT: .amdhsa_system_sgpr_workgroup_info 0
; CHECK-NEXT: .amdhsa_system_vgpr_workitem_id 0
define amdgpu_kernel void @indirect_call_known_no_special_inputs() {
bb:
  %cond = load i1, i1 addrspace(4)* null
  %tmp = select i1 %cond, void (i8*, i32, i8*)* bitcast (void ()* @wobble to void (i8*, i32, i8*)*), void (i8*, i32, i8*)* bitcast (void ()* @snork to void (i8*, i32, i8*)*)
  call void %tmp(i8* undef, i32 undef, i8* undef)
  ret void
}

define void @wobble() {
bb:
  ret void
}

define void @snork() {
bb:
  ret void
}
