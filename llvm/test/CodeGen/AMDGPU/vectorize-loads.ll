; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}load_idx_idy:
; GCN-NOT: global_load
; GCN: s_load_dword [[ID_XY:s[0-9]+]], s[4:5], 0x4
; GCN-NOT: global_load
; GCN: s_lshr_b32 [[ID_Y:s[0-9]+]], [[ID_XY]], 16
; GCN: s_add_i32 [[ID_SUM:s[0-9]+]], [[ID_Y]], [[ID_XY]]
; GCN: s_and_b32 s{{[0-9]+}}, [[ID_SUM]], 0xffff
define protected amdgpu_kernel void @load_idx_idy(i32 addrspace(1)* %out) {
entry:
  %disp = tail call align 4 dereferenceable(64) i8 addrspace(4)* @llvm.amdgcn.dispatch.ptr()
  %gep_x = getelementptr i8, i8 addrspace(4)* %disp, i64 4
  %gep_x.cast = bitcast i8 addrspace(4)* %gep_x to i16 addrspace(4)*
  %id_x = load i16, i16 addrspace(4)* %gep_x.cast, align 4, !invariant.load !0 ; load workgroup size x
  %gep_y = getelementptr i8, i8 addrspace(4)* %disp, i64 6
  %gep_y.cast = bitcast i8 addrspace(4)* %gep_y to i16 addrspace(4)*
  %id_y = load i16, i16 addrspace(4)* %gep_y.cast, align 2, !invariant.load !0 ; load workgroup size y
  %add = add nuw nsw i16 %id_y, %id_x
  %conv = zext i16 %add to i32
  store i32 %conv, i32 addrspace(1)* %out, align 4
  ret void
}

; A little more complicated case where more sub-dword loads could be coalesced
; if they are not widening earlier.
; GCN-LABEL: {{^}}load_4i16:
; GCN: s_load_dwordx2 s[[[D0:[0-9]+]]:[[D1:[0-9]+]]], s[4:5], 0x4
; GCN-NOT: s_load_dword {{s[0-9]+}}, s[4:5], 0x4
; GCN-DAG: s_lshr_b32 s{{[0-9]+}}, s[[D0]], 16
; GCN-DAG: s_lshr_b32 s{{[0-9]+}}, s[[D1]], 16
; GCN: s_endpgm
define protected amdgpu_kernel void @load_4i16(i32 addrspace(1)* %out) {
entry:
  %disp = tail call align 4 dereferenceable(64) i8 addrspace(4)* @llvm.amdgcn.dispatch.ptr()
  %gep_x = getelementptr i8, i8 addrspace(4)* %disp, i64 4
  %gep_x.cast = bitcast i8 addrspace(4)* %gep_x to i16 addrspace(4)*
  %id_x = load i16, i16 addrspace(4)* %gep_x.cast, align 4, !invariant.load !0 ; load workgroup size x
  %gep_y = getelementptr i8, i8 addrspace(4)* %disp, i64 6
  %gep_y.cast = bitcast i8 addrspace(4)* %gep_y to i16 addrspace(4)*
  %id_y = load i16, i16 addrspace(4)* %gep_y.cast, align 2, !invariant.load !0 ; load workgroup size y
  %gep_z = getelementptr i8, i8 addrspace(4)* %disp, i64 8
  %gep_z.cast = bitcast i8 addrspace(4)* %gep_z to i16 addrspace(4)*
  %id_z = load i16, i16 addrspace(4)* %gep_z.cast, align 4, !invariant.load !0 ; load workgroup size x
  %gep_w = getelementptr i8, i8 addrspace(4)* %disp, i64 10
  %gep_w.cast = bitcast i8 addrspace(4)* %gep_w to i16 addrspace(4)*
  %id_w = load i16, i16 addrspace(4)* %gep_w.cast, align 2, !invariant.load !0 ; load workgroup size y
  %add = add nuw nsw i16 %id_y, %id_x
  %add2 = add nuw nsw i16 %id_z, %id_w
  %add3 = add nuw nsw i16 %add, %add2
  %conv = zext i16 %add3 to i32
  store i32 %conv, i32 addrspace(1)* %out, align 4
  ret void
}

declare i8 addrspace(4)* @llvm.amdgcn.dispatch.ptr()

!0 = !{!0}
