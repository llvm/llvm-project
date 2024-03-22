; RUN: llc -mtriple=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

declare i32 @llvm.amdgcn.sad.u8(i32, i32, i32) nounwind readnone

; GCN-LABEL: {{^}}v_sad_u8:
; GCN: v_sad_u8 v{{[0-9]+}}, v{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}
define amdgpu_kernel void @v_sad_u8(ptr addrspace(1) %out, i32 %src) {
  %result= call i32 @llvm.amdgcn.sad.u8(i32 %src, i32 100, i32 100) nounwind readnone
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}v_sad_u8_non_immediate:
; GCN: v_sad_u8 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @v_sad_u8_non_immediate(ptr addrspace(1) %out, i32 %src, i32 %a, i32 %b) {
  %result= call i32 @llvm.amdgcn.sad.u8(i32 %src, i32 %a, i32 %b) nounwind readnone
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}
