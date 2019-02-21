; RUN: llc -mtriple=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX8 %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX9 %s

; GCN-LABEL: reassoc_i32:
; GCN: s_add_i32 [[ADD1:s[0-9]+]], s{{[0-9]+}}, s{{[0-9]+}}
; GFX8: v_add_u32_e32 v{{[0-9]+}}, vcc, [[ADD1]], v{{[0-9]+}}
; GFX9: v_add_u32_e32 v{{[0-9]+}}, [[ADD1]], v{{[0-9]+}}
define amdgpu_kernel void @reassoc_i32(i32 addrspace(1)* %arg, i32 %x, i32 %y) {
bb:
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %add1 = add i32 %x, %tid
  %add2 = add i32 %add1, %y
  store i32 %add2, i32 addrspace(1)* %arg, align 4
  ret void
}

; GCN-LABEL: reassoc_i32_swap_arg_order:
; GCN:  s_add_i32 [[ADD1:s[0-9]+]], s{{[0-9]+}}, s{{[0-9]+}}
; GFX8: v_add_u32_e32 v{{[0-9]+}}, vcc, [[ADD1]], v{{[0-9]+}}
; GFX9: v_add_u32_e32 v{{[0-9]+}}, [[ADD1]], v{{[0-9]+}}
define amdgpu_kernel void @reassoc_i32_swap_arg_order(i32 addrspace(1)* %arg, i32 %x, i32 %y) {
bb:
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %add1 = add i32 %tid, %x
  %add2 = add i32 %y, %add1
  store i32 %add2, i32 addrspace(1)* %arg, align 4
  ret void
}

; GCN-LABEL: reassoc_i64:
; GCN:      s_add_u32 [[ADD1L:s[0-9]+]], s{{[0-9]+}}, s{{[0-9]+}}
; GCN:      s_addc_u32 [[ADD1H:s[0-9]+]], s{{[0-9]+}}, s{{[0-9]+}}
; GFX8-DAG: v_add_u32_e32 v{{[0-9]+}}, vcc, [[ADD1L]], v{{[0-9]+}}
; GFX9-DAG: v_add_co_u32_e32 v{{[0-9]+}}, vcc, [[ADD1L]], v{{[0-9]+}}
; GCN-DAG:  v_mov_b32_e32 [[VADD1H:v[0-9]+]], [[ADD1H]]
; GFX8:     v_addc_u32_e32 v{{[0-9]+}}, vcc, 0, [[VADD1H]], vcc
; GFX9:     v_addc_co_u32_e32 v{{[0-9]+}}, vcc, 0, [[VADD1H]], vcc
define amdgpu_kernel void @reassoc_i64(i64 addrspace(1)* %arg, i64 %x, i64 %y) {
bb:
  %tid32 = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tid = zext i32 %tid32 to i64
  %add1 = add i64 %x, %tid
  %add2 = add i64 %add1, %y
  store i64 %add2, i64 addrspace(1)* %arg, align 8
  ret void
}

; GCN-LABEL: reassoc_v2i32:
; GCN: s_add_i32 [[ADD1:s[0-9]+]], s{{[0-9]+}}, s{{[0-9]+}}
; GCN: s_add_i32 [[ADD2:s[0-9]+]], s{{[0-9]+}}, s{{[0-9]+}}
; GFX8: v_add_u32_e32 v{{[0-9]+}}, vcc, [[ADD1]], v{{[0-9]+}}
; GFX8: v_add_u32_e32 v{{[0-9]+}}, vcc, [[ADD2]], v{{[0-9]+}}
; GFX9: v_add_u32_e32 v{{[0-9]+}}, [[ADD1]], v{{[0-9]+}}
; GFX9: v_add_u32_e32 v{{[0-9]+}}, [[ADD2]], v{{[0-9]+}}
define amdgpu_kernel void @reassoc_v2i32(<2 x i32> addrspace(1)* %arg, <2 x i32> %x, <2 x i32> %y) {
bb:
  %t1 = tail call i32 @llvm.amdgcn.workitem.id.x()
  %t2 = tail call i32 @llvm.amdgcn.workitem.id.y()
  %v1 = insertelement <2 x i32> undef, i32 %t1, i32 0
  %v2 = insertelement <2 x i32> %v1, i32 %t2, i32 1
  %add1 = add <2 x i32> %x, %v2
  %add2 = add <2 x i32> %add1, %y
  store <2 x i32> %add2, <2 x i32> addrspace(1)* %arg, align 4
  ret void
}

; GCN-LABEL: reassoc_i32_nuw:
; GCN:  s_add_i32 [[ADD1:s[0-9]+]], s{{[0-9]+}}, s{{[0-9]+}}
; GFX8: v_add_u32_e32 v{{[0-9]+}}, vcc, [[ADD1]], v{{[0-9]+}}
; GFX9: v_add_u32_e32 v{{[0-9]+}}, [[ADD1]], v{{[0-9]+}}
define amdgpu_kernel void @reassoc_i32_nuw(i32 addrspace(1)* %arg, i32 %x, i32 %y) {
bb:
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %add1 = add i32 %x, %tid
  %add2 = add nuw i32 %add1, %y
  store i32 %add2, i32 addrspace(1)* %arg, align 4
  ret void
}

; GCN-LABEL: reassoc_i32_multiuse:
; GFX8: v_add_u32_e32 [[ADD1:v[0-9]+]], vcc, s{{[0-9]+}}, v{{[0-9]+}}
; GFX9: v_add_u32_e32 [[ADD1:v[0-9]+]], s{{[0-9]+}}, v{{[0-9]+}}
; GFX8: v_add_u32_e32 v{{[0-9]+}}, vcc, s{{[0-9]+}}, [[ADD1]]
; GFX9: v_add_u32_e32 v{{[0-9]+}}, s{{[0-9]+}}, [[ADD1]]
define amdgpu_kernel void @reassoc_i32_multiuse(i32 addrspace(1)* %arg, i32 %x, i32 %y) {
bb:
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %add1 = add i32 %x, %tid
  %add2 = add i32 %add1, %y
  store volatile i32 %add1, i32 addrspace(1)* %arg, align 4
  store volatile i32 %add2, i32 addrspace(1)* %arg, align 4
  ret void
}

; TODO: This should be reassociated as well, however it is disabled to avoid endless
;       loop since DAGCombiner::ReassociateOps() reverts the reassociation.
; GCN-LABEL: reassoc_i32_const:
; GFX8: v_add_u32_e32 [[ADD1:v[0-9]+]], vcc, 42, v{{[0-9]+}}
; GFX9: v_add_u32_e32 [[ADD1:v[0-9]+]],  42, v{{[0-9]+}}
; GFX8: v_add_u32_e32 v{{[0-9]+}}, vcc, s{{[0-9]+}}, [[ADD1]]
; GFX9: v_add_u32_e32 v{{[0-9]+}}, s{{[0-9]+}}, [[ADD1]]
define amdgpu_kernel void @reassoc_i32_const(i32 addrspace(1)* %arg, i32 %x) {
bb:
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %add1 = add i32 %tid, 42
  %add2 = add i32 %add1, %x
  store volatile i32 %add1, i32 addrspace(1)* %arg, align 4
  store volatile i32 %add2, i32 addrspace(1)* %arg, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
declare i32 @llvm.amdgcn.workitem.id.y()
