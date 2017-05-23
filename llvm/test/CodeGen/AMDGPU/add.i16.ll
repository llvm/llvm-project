; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=VI -check-prefix=GCN %s

; FIXME: Need to handle non-uniform case for function below (load without gep).
; GCN-LABEL: {{^}}v_test_add_i16:
; VI: flat_load_ushort [[A:v[0-9]+]]
; VI: flat_load_ushort [[B:v[0-9]+]]
; VI: v_add_u16_e32 [[ADD:v[0-9]+]], [[B]], [[A]]
; VI-NEXT: buffer_store_short [[ADD]]
define amdgpu_kernel void @v_test_add_i16(i16 addrspace(1)* %out, i16 addrspace(1)* %in0, i16 addrspace(1)* %in1) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.out = getelementptr inbounds i16, i16 addrspace(1)* %out, i32 %tid
  %gep.in0 = getelementptr inbounds i16, i16 addrspace(1)* %in0, i32 %tid
  %gep.in1 = getelementptr inbounds i16, i16 addrspace(1)* %in1, i32 %tid
  %a = load volatile i16, i16 addrspace(1)* %gep.in0
  %b = load volatile i16, i16 addrspace(1)* %gep.in1
  %add = add i16 %a, %b
  store i16 %add, i16 addrspace(1)* %out
  ret void
}

; FIXME: Need to handle non-uniform case for function below (load without gep).
; GCN-LABEL: {{^}}v_test_add_i16_constant:
; VI: flat_load_ushort [[A:v[0-9]+]]
; VI: v_add_u16_e32 [[ADD:v[0-9]+]], 0x7b, [[A]]
; VI-NEXT: buffer_store_short [[ADD]]
define amdgpu_kernel void @v_test_add_i16_constant(i16 addrspace(1)* %out, i16 addrspace(1)* %in0) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.out = getelementptr inbounds i16, i16 addrspace(1)* %out, i32 %tid
  %gep.in0 = getelementptr inbounds i16, i16 addrspace(1)* %in0, i32 %tid
  %a = load volatile i16, i16 addrspace(1)* %gep.in0
  %add = add i16 %a, 123
  store i16 %add, i16 addrspace(1)* %out
  ret void
}

; FIXME: Need to handle non-uniform case for function below (load without gep).
; GCN-LABEL: {{^}}v_test_add_i16_neg_constant:
; VI: flat_load_ushort [[A:v[0-9]+]]
; VI: v_add_u16_e32 [[ADD:v[0-9]+]], 0xfffffcb3, [[A]]
; VI-NEXT: buffer_store_short [[ADD]]
define amdgpu_kernel void @v_test_add_i16_neg_constant(i16 addrspace(1)* %out, i16 addrspace(1)* %in0) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.out = getelementptr inbounds i16, i16 addrspace(1)* %out, i32 %tid
  %gep.in0 = getelementptr inbounds i16, i16 addrspace(1)* %in0, i32 %tid
  %a = load volatile i16, i16 addrspace(1)* %gep.in0
  %add = add i16 %a, -845
  store i16 %add, i16 addrspace(1)* %out
  ret void
}

; FIXME: Need to handle non-uniform case for function below (load without gep).
; GCN-LABEL: {{^}}v_test_add_i16_inline_neg1:
; VI: flat_load_ushort [[A:v[0-9]+]]
; VI: v_add_u16_e32 [[ADD:v[0-9]+]], -1, [[A]]
; VI-NEXT: buffer_store_short [[ADD]]
define amdgpu_kernel void @v_test_add_i16_inline_neg1(i16 addrspace(1)* %out, i16 addrspace(1)* %in0) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.out = getelementptr inbounds i16, i16 addrspace(1)* %out, i32 %tid
  %gep.in0 = getelementptr inbounds i16, i16 addrspace(1)* %in0, i32 %tid
  %a = load volatile i16, i16 addrspace(1)* %gep.in0
  %add = add i16 %a, -1
  store i16 %add, i16 addrspace(1)* %out
  ret void
}

; FIXME: Need to handle non-uniform case for function below (load without gep).
; GCN-LABEL: {{^}}v_test_add_i16_zext_to_i32:
; VI: flat_load_ushort [[A:v[0-9]+]]
; VI: flat_load_ushort [[B:v[0-9]+]]
; VI: v_add_u16_e32 [[ADD:v[0-9]+]], [[B]], [[A]]
; VI-NEXT: buffer_store_dword [[ADD]]
define amdgpu_kernel void @v_test_add_i16_zext_to_i32(i32 addrspace(1)* %out, i16 addrspace(1)* %in0, i16 addrspace(1)* %in1) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.out = getelementptr inbounds i32, i32 addrspace(1)* %out, i32 %tid
  %gep.in0 = getelementptr inbounds i16, i16 addrspace(1)* %in0, i32 %tid
  %gep.in1 = getelementptr inbounds i16, i16 addrspace(1)* %in1, i32 %tid
  %a = load volatile i16, i16 addrspace(1)* %gep.in0
  %b = load volatile i16, i16 addrspace(1)* %gep.in1
  %add = add i16 %a, %b
  %ext = zext i16 %add to i32
  store i32 %ext, i32 addrspace(1)* %out
  ret void
}

; FIXME: Need to handle non-uniform case for function below (load without gep).
; GCN-LABEL: {{^}}v_test_add_i16_zext_to_i64:
; VI: flat_load_ushort [[A:v[0-9]+]]
; VI: flat_load_ushort [[B:v[0-9]+]]
; VI-DAG: v_add_u16_e32 v[[ADD:[0-9]+]], [[B]], [[A]]
; VI: buffer_store_dwordx2 v{{\[}}[[ADD]]:{{[0-9]+\]}}, off, {{s\[[0-9]+:[0-9]+\]}}, 0{{$}}
define amdgpu_kernel void @v_test_add_i16_zext_to_i64(i64 addrspace(1)* %out, i16 addrspace(1)* %in0, i16 addrspace(1)* %in1) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.out = getelementptr inbounds i64, i64 addrspace(1)* %out, i32 %tid
  %gep.in0 = getelementptr inbounds i16, i16 addrspace(1)* %in0, i32 %tid
  %gep.in1 = getelementptr inbounds i16, i16 addrspace(1)* %in1, i32 %tid
  %a = load volatile i16, i16 addrspace(1)* %gep.in0
  %b = load volatile i16, i16 addrspace(1)* %gep.in1
  %add = add i16 %a, %b
  %ext = zext i16 %add to i64
  store i64 %ext, i64 addrspace(1)* %out
  ret void
}

; FIXME: Need to handle non-uniform case for function below (load without gep).
; GCN-LABEL: {{^}}v_test_add_i16_sext_to_i32:
; VI: flat_load_ushort [[A:v[0-9]+]]
; VI: flat_load_ushort [[B:v[0-9]+]]
; VI: v_add_u16_e32 [[ADD:v[0-9]+]],  [[B]], [[A]]
; VI-NEXT: v_bfe_i32 [[SEXT:v[0-9]+]], [[ADD]], 0, 16
; VI-NEXT: buffer_store_dword [[SEXT]]
define amdgpu_kernel void @v_test_add_i16_sext_to_i32(i32 addrspace(1)* %out, i16 addrspace(1)* %in0, i16 addrspace(1)* %in1) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.out = getelementptr inbounds i32, i32 addrspace(1)* %out, i32 %tid
  %gep.in0 = getelementptr inbounds i16, i16 addrspace(1)* %in0, i32 %tid
  %gep.in1 = getelementptr inbounds i16, i16 addrspace(1)* %in1, i32 %tid
  %a = load i16, i16 addrspace(1)* %gep.in0
  %b = load i16, i16 addrspace(1)* %gep.in1
  %add = add i16 %a, %b
  %ext = sext i16 %add to i32
  store i32 %ext, i32 addrspace(1)* %out
  ret void
}

; FIXME: Need to handle non-uniform case for function below (load without gep).
; GCN-LABEL: {{^}}v_test_add_i16_sext_to_i64:
; VI: flat_load_ushort [[A:v[0-9]+]]
; VI: flat_load_ushort [[B:v[0-9]+]]
; VI: v_add_u16_e32 [[ADD:v[0-9]+]], [[B]], [[A]]
; VI-NEXT: v_bfe_i32 v[[LO:[0-9]+]], [[ADD]], 0, 16
; VI-NEXT: v_ashrrev_i32_e32 v[[HI:[0-9]+]], 31, v[[LO]]
; VI-NEXT: buffer_store_dwordx2 v{{\[}}[[LO]]:[[HI]]{{\]}}
define amdgpu_kernel void @v_test_add_i16_sext_to_i64(i64 addrspace(1)* %out, i16 addrspace(1)* %in0, i16 addrspace(1)* %in1) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.out = getelementptr inbounds i64, i64 addrspace(1)* %out, i32 %tid
  %gep.in0 = getelementptr inbounds i16, i16 addrspace(1)* %in0, i32 %tid
  %gep.in1 = getelementptr inbounds i16, i16 addrspace(1)* %in1, i32 %tid
  %a = load i16, i16 addrspace(1)* %gep.in0
  %b = load i16, i16 addrspace(1)* %gep.in1
  %add = add i16 %a, %b
  %ext = sext i16 %add to i64
  store i64 %ext, i64 addrspace(1)* %out
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #0

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
