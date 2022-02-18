; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=VI -check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=bonaire -verify-machineinstrs < %s | FileCheck -check-prefix=CI -check-prefix=GCN %s

; FIXME: Need to handle non-uniform case for function below (load without gep).
; GCN-LABEL: {{^}}v_test_sub_i16:
; VI: flat_load_ushort [[A:v[0-9]+]]
; VI: flat_load_ushort [[B:v[0-9]+]]
; VI: v_sub_u16_e32 [[ADD:v[0-9]+]], [[A]], [[B]]
; VI-NEXT: buffer_store_short [[ADD]]
define amdgpu_kernel void @v_test_sub_i16(i16 addrspace(1)* %out, i16 addrspace(1)* %in0, i16 addrspace(1)* %in1) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.out = getelementptr inbounds i16, i16 addrspace(1)* %out, i32 %tid
  %gep.in0 = getelementptr inbounds i16, i16 addrspace(1)* %in0, i32 %tid
  %gep.in1 = getelementptr inbounds i16, i16 addrspace(1)* %in1, i32 %tid
  %a = load volatile i16, i16 addrspace(1)* %gep.in0
  %b = load volatile i16, i16 addrspace(1)* %gep.in1
  %add = sub i16 %a, %b
  store i16 %add, i16 addrspace(1)* %out
  ret void
}

; FIXME: Need to handle non-uniform case for function below (load without gep).
; GCN-LABEL: {{^}}v_test_sub_i16_constant:
; VI: flat_load_ushort [[A:v[0-9]+]]
; VI: v_add_u16_e32 [[ADD:v[0-9]+]], 0xff85, [[A]]
; VI-NEXT: buffer_store_short [[ADD]]
define amdgpu_kernel void @v_test_sub_i16_constant(i16 addrspace(1)* %out, i16 addrspace(1)* %in0) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.out = getelementptr inbounds i16, i16 addrspace(1)* %out, i32 %tid
  %gep.in0 = getelementptr inbounds i16, i16 addrspace(1)* %in0, i32 %tid
  %a = load volatile i16, i16 addrspace(1)* %gep.in0
  %add = sub i16 %a, 123
  store i16 %add, i16 addrspace(1)* %out
  ret void
}

; FIXME: Need to handle non-uniform case for function below (load without gep).
; GCN-LABEL: {{^}}v_test_sub_i16_neg_constant:
; VI: flat_load_ushort [[A:v[0-9]+]]
; VI: v_add_u16_e32 [[ADD:v[0-9]+]], 0x34d, [[A]]
; VI-NEXT: buffer_store_short [[ADD]]
define amdgpu_kernel void @v_test_sub_i16_neg_constant(i16 addrspace(1)* %out, i16 addrspace(1)* %in0) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.out = getelementptr inbounds i16, i16 addrspace(1)* %out, i32 %tid
  %gep.in0 = getelementptr inbounds i16, i16 addrspace(1)* %in0, i32 %tid
  %a = load volatile i16, i16 addrspace(1)* %gep.in0
  %add = sub i16 %a, -845
  store i16 %add, i16 addrspace(1)* %out
  ret void
}

; FIXME: Need to handle non-uniform case for function below (load without gep).
; GCN-LABEL: {{^}}v_test_sub_i16_inline_63:
; VI: flat_load_ushort [[A:v[0-9]+]]
; VI: v_subrev_u16_e32 [[ADD:v[0-9]+]], 63, [[A]]
; VI-NEXT: buffer_store_short [[ADD]]
define amdgpu_kernel void @v_test_sub_i16_inline_63(i16 addrspace(1)* %out, i16 addrspace(1)* %in0) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.out = getelementptr inbounds i16, i16 addrspace(1)* %out, i32 %tid
  %gep.in0 = getelementptr inbounds i16, i16 addrspace(1)* %in0, i32 %tid
  %a = load volatile i16, i16 addrspace(1)* %gep.in0
  %add = sub i16 %a, 63
  store i16 %add, i16 addrspace(1)* %out
  ret void
}

; FIXME: Need to handle non-uniform case for function below (load without gep).
; GCN-LABEL: {{^}}v_test_sub_i16_zext_to_i32:
; VI: flat_load_ushort [[A:v[0-9]+]]
; VI: flat_load_ushort [[B:v[0-9]+]]
; VI: v_sub_u16_e32 [[ADD:v[0-9]+]], [[A]], [[B]]
; VI-NEXT: buffer_store_dword [[ADD]]
define amdgpu_kernel void @v_test_sub_i16_zext_to_i32(i32 addrspace(1)* %out, i16 addrspace(1)* %in0, i16 addrspace(1)* %in1) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.out = getelementptr inbounds i32, i32 addrspace(1)* %out, i32 %tid
  %gep.in0 = getelementptr inbounds i16, i16 addrspace(1)* %in0, i32 %tid
  %gep.in1 = getelementptr inbounds i16, i16 addrspace(1)* %in1, i32 %tid
  %a = load volatile i16, i16 addrspace(1)* %gep.in0
  %b = load volatile i16, i16 addrspace(1)* %gep.in1
  %add = sub i16 %a, %b
  %ext = zext i16 %add to i32
  store i32 %ext, i32 addrspace(1)* %out
  ret void
}

; FIXME: Need to handle non-uniform case for function below (load without gep).
; GCN-LABEL: {{^}}v_test_sub_i16_zext_to_i64:
; VI: flat_load_ushort [[A:v[0-9]+]]
; VI: flat_load_ushort [[B:v[0-9]+]]
; VI: v_mov_b32_e32 v[[VZERO:[0-9]+]], 0
; VI-DAG: v_sub_u16_e32 v[[ADD:[0-9]+]], [[A]], [[B]]
; VI: buffer_store_dwordx2 v[[[ADD]]:[[VZERO]]], off, {{s\[[0-9]+:[0-9]+\]}}, 0{{$}}
define amdgpu_kernel void @v_test_sub_i16_zext_to_i64(i64 addrspace(1)* %out, i16 addrspace(1)* %in0, i16 addrspace(1)* %in1) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.out = getelementptr inbounds i64, i64 addrspace(1)* %out, i32 %tid
  %gep.in0 = getelementptr inbounds i16, i16 addrspace(1)* %in0, i32 %tid
  %gep.in1 = getelementptr inbounds i16, i16 addrspace(1)* %in1, i32 %tid
  %a = load volatile i16, i16 addrspace(1)* %gep.in0
  %b = load volatile i16, i16 addrspace(1)* %gep.in1
  %add = sub i16 %a, %b
  %ext = zext i16 %add to i64
  store i64 %ext, i64 addrspace(1)* %out
  ret void
}

; FIXME: Need to handle non-uniform case for function below (load without gep).
; GCN-LABEL: {{^}}v_test_sub_i16_sext_to_i32:
; VI: flat_load_ushort [[A:v[0-9]+]]
; VI: flat_load_ushort [[B:v[0-9]+]]
; VI: v_sub_u16_e32 [[ADD:v[0-9]+]], [[A]], [[B]]
; VI: v_bfe_i32 [[SEXT:v[0-9]+]], [[ADD]], 0, 16
; VI-NEXT: buffer_store_dword [[SEXT]]
define amdgpu_kernel void @v_test_sub_i16_sext_to_i32(i32 addrspace(1)* %out, i16 addrspace(1)* %in0, i16 addrspace(1)* %in1) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.out = getelementptr inbounds i32, i32 addrspace(1)* %out, i32 %tid
  %gep.in0 = getelementptr inbounds i16, i16 addrspace(1)* %in0, i32 %tid
  %gep.in1 = getelementptr inbounds i16, i16 addrspace(1)* %in1, i32 %tid
  %a = load i16, i16 addrspace(1)* %gep.in0
  %b = load i16, i16 addrspace(1)* %gep.in1
  %add = sub i16 %a, %b
  %ext = sext i16 %add to i32
  store i32 %ext, i32 addrspace(1)* %out
  ret void
}

; FIXME: Need to handle non-uniform case for function below (load without gep).
; GCN-LABEL: {{^}}v_test_sub_i16_sext_to_i64:
; VI: flat_load_ushort [[A:v[0-9]+]]
; VI: flat_load_ushort [[B:v[0-9]+]]
; VI: v_sub_u16_e32 [[ADD:v[0-9]+]], [[A]], [[B]]
; VI-NEXT: v_bfe_i32 v[[LO:[0-9]+]], [[ADD]], 0, 16
; VI:      v_ashrrev_i32_e32 v[[HI:[0-9]+]], 31, v[[LO]]
; VI-NEXT: buffer_store_dwordx2 v[[[LO]]:[[HI]]]
define amdgpu_kernel void @v_test_sub_i16_sext_to_i64(i64 addrspace(1)* %out, i16 addrspace(1)* %in0, i16 addrspace(1)* %in1) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.out = getelementptr inbounds i64, i64 addrspace(1)* %out, i32 %tid
  %gep.in0 = getelementptr inbounds i16, i16 addrspace(1)* %in0, i32 %tid
  %gep.in1 = getelementptr inbounds i16, i16 addrspace(1)* %in1, i32 %tid
  %a = load i16, i16 addrspace(1)* %gep.in0
  %b = load i16, i16 addrspace(1)* %gep.in1
  %add = sub i16 %a, %b
  %ext = sext i16 %add to i64
  store i64 %ext, i64 addrspace(1)* %out
  ret void
}

@lds = addrspace(3) global [512 x i32] undef, align 4

; GCN-LABEL: {{^}}v_test_sub_i16_constant_commute:
; VI: v_subrev_u16_e32 v{{[0-9]+}}, 0x800, v{{[0-9]+}}
; CI: v_subrev_i32_e32 v{{[0-9]+}}, vcc, 0x800, v{{[0-9]+}}
define amdgpu_kernel void @v_test_sub_i16_constant_commute(i16 addrspace(1)* %out, i16 addrspace(1)* %in0) #1 {
  %size = call i32 @llvm.amdgcn.groupstaticsize()
  %size.trunc = trunc i32 %size to i16
  call void asm sideeffect "; $0", "v"([512 x i32] addrspace(3)* @lds)
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.out = getelementptr inbounds i16, i16 addrspace(1)* %out, i32 %tid
  %gep.in0 = getelementptr inbounds i16, i16 addrspace(1)* %in0, i32 %tid
  %a = load volatile i16, i16 addrspace(1)* %gep.in0
  %add = sub i16 %a, %size.trunc
  store i16 %add, i16 addrspace(1)* %out
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #0
declare i32 @llvm.amdgcn.groupstaticsize() #0

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
