; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: @bfe_uniform
; GCN: s_bfe_u32 s{{[0-9]+}}, s{{[0-9]+}}, 0x40010
define amdgpu_kernel void @bfe_uniform(i32 %val, ptr addrspace(1) %out) {
  %hibits = lshr i32 %val, 16
  %masked = and i32 %hibits, 15
  store i32 %masked, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: @bfe_divergent
; GCN: v_bfe_u32 v{{[0-9]+}}, v{{[0-9]+}}, 16, 4
define amdgpu_kernel void @bfe_divergent(i32 %val, ptr addrspace(1) %out) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %divergent = add i32 %val, %tid
  %hibits = lshr i32 %divergent, 16
  %masked = and i32 %hibits, 15
  store i32 %masked, ptr addrspace(1) %out
  ret void
}


declare i32 @llvm.amdgcn.workitem.id.x()

