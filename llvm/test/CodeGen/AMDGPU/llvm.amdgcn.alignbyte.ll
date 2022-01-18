; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

declare i32 @llvm.amdgcn.alignbyte(i32, i32, i32) #0

; GCN-LABEL: {{^}}v_alignbyte_b32:
; GCN: v_alignbyte_b32 {{[vs][0-9]+}}, {{[vs][0-9]+}}, {{[vs][0-9]+}}
define amdgpu_kernel void @v_alignbyte_b32(i32 addrspace(1)* %out, i32 %src1, i32 %src2, i32 %src3) #1 {
  %val = call i32 @llvm.amdgcn.alignbyte(i32 %src1, i32 %src2, i32 %src3) #0
  store i32 %val, i32 addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
