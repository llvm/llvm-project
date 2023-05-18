; RUN: llc -mtriple=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}and_i1_sext_bool:
; GCN: v_cmp_{{gt|le}}_u32_e{{32|64}} [[CC:[^,]+]], v{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_cndmask_b32_e{{32|64}} [[VAL:v[0-9]+]], 0, v{{[0-9]+}}, [[CC]]
; GCN: store_dword {{.*}}[[VAL]]
; GCN-NOT: v_cndmask_b32_e64 v{{[0-9]+}}, {{0|-1}}, {{0|-1}}
; GCN-NOT: v_and_b32_e32

define amdgpu_kernel void @and_i1_sext_bool(ptr addrspace(1) nocapture %arg) {
bb:
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %y = tail call i32 @llvm.amdgcn.workitem.id.y()
  %gep = getelementptr inbounds i32, ptr addrspace(1) %arg, i32 %x
  %v = load i32, ptr addrspace(1) %gep, align 4
  %cmp = icmp ugt i32 %x, %y
  %ext = sext i1 %cmp to i32
  %and = and i32 %v, %ext
  store i32 %and, ptr addrspace(1) %gep, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #0

declare i32 @llvm.amdgcn.workitem.id.y() #0

attributes #0 = { nounwind readnone speculatable }
