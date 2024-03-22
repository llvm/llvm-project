; RUN: llc -mtriple=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

declare i32 @llvm.amdgcn.alignbyte(i32, i32, i32) nounwind readnone

; GCN-LABEL: {{^}}v_alignbyte_b32:
; GCN: v_alignbyte_b32 {{[vs][0-9]+}}, {{[vs][0-9]+}}, {{[vs][0-9]+}}
define amdgpu_kernel void @v_alignbyte_b32(ptr addrspace(1) %out, i32 %src1, i32 %src2, i32 %src3) nounwind {
  %val = call i32 @llvm.amdgcn.alignbyte(i32 %src1, i32 %src2, i32 %src3) nounwind readnone
  store i32 %val, ptr addrspace(1) %out
  ret void
}
