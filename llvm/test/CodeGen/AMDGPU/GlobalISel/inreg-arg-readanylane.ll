; RUN: llc -global-isel -mtriple=amdgpu9.00-mesa-mesa3d -o - %s | FileCheck %s

; CHECK-LABEL: inreg_arg_in_vgpr:
; CHECK-NOT: v_readfirstlane_b32
; CHECK: global_load_dword
; CHECK-NOT: v_readfirstlane_b32
; CHECK: v_add_u32
; CHECK-NOT: v_readfirstlane_b32
; CHECK: global_store_dword
; CHECK-NOT: v_readfirstlane_b32

define void @inreg_arg_in_vgpr([28 x i32] inreg %pad, i32 inreg %x, ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p
  %a = add i32 %v, %x
  store i32 %a, ptr addrspace(1) %p
  ret void
}