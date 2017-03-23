; RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck %s --check-prefix=GCN

; GCN-LABEL: and_zext:
; GCN: v_and_b32_e32 [[VAL16:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_and_b32_e32 v{{[0-9]+}}, 0xffff, [[VAL16]]
define amdgpu_kernel void @and_zext(i32 addrspace(1)* %out, i16 addrspace(1)* %in) {
  %id = call i32 @llvm.amdgcn.workitem.id.x() #1
  %ptr = getelementptr i16, i16 addrspace(1)* %in, i32 %id
  %a = load i16, i16 addrspace(1)* %in
  %b = load i16, i16 addrspace(1)* %ptr
  %c = add i16 %a, %b
  %val16 = and i16 %c, %a
  %val32 = zext i16 %val16 to i32
  store i32 %val32, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: or_zext:
; GCN: v_or_b32_e32 [[VAL16:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_and_b32_e32 v{{[0-9]+}}, 0xffff, [[VAL16]]
define amdgpu_kernel void @or_zext(i32 addrspace(1)* %out, i16 addrspace(1)* %in) {
  %id = call i32 @llvm.amdgcn.workitem.id.x() #1
  %ptr = getelementptr i16, i16 addrspace(1)* %in, i32 %id
  %a = load i16, i16 addrspace(1)* %in
  %b = load i16, i16 addrspace(1)* %ptr
  %c = add i16 %a, %b
  %val16 = or i16 %c, %a
  %val32 = zext i16 %val16 to i32
  store i32 %val32, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: xor_zext:
; GCN: v_xor_b32_e32 [[VAL16:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_and_b32_e32 v{{[0-9]+}}, 0xffff, [[VAL16]]
define amdgpu_kernel void @xor_zext(i32 addrspace(1)* %out, i16 addrspace(1)* %in) {
  %id = call i32 @llvm.amdgcn.workitem.id.x() #1
  %ptr = getelementptr i16, i16 addrspace(1)* %in, i32 %id
  %a = load i16, i16 addrspace(1)* %in
  %b = load i16, i16 addrspace(1)* %ptr
  %c = add i16 %a, %b
  %val16 = xor i16 %c, %a
  %val32 = zext i16 %val16 to i32
  store i32 %val32, i32 addrspace(1)* %out
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #1 = { nounwind readnone }
