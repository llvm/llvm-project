; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

declare i32 @llvm.amdgcn.lerp(i32, i32, i32) #0

; GCN-LABEL: {{^}}v_lerp:
; GCN: v_lerp_u8 v{{[0-9]+}}, v{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}
define amdgpu_kernel void @v_lerp(i32 addrspace(1)* %out, i32 %src) nounwind {
  %result= call i32 @llvm.amdgcn.lerp(i32 %src, i32 100, i32 100) #0
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

attributes #0 = { nounwind readnone }
