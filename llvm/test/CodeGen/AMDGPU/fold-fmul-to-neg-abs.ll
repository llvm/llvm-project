; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}fold_mul_neg:
; GCN: load_dword [[V:v[0-9]+]]
; GCN: v_or_b32_e32 [[NEG:v[0-9]]], 0x80000000, [[V]]
; GCN: store_dword [[NEG]]

define amdgpu_kernel void @fold_mul_neg(ptr addrspace(1) %arg) {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, ptr addrspace(1) %arg, i32 %tid
  %v = load float, ptr addrspace(1) %gep, align 4
  %cmp = fcmp fast ogt float %v, 0.000000e+00
  %sel = select i1 %cmp, float -1.000000e+00, float 1.000000e+00
  %mul = fmul fast float %v, %sel
  store float %mul, ptr addrspace(1) %gep, align 4
  ret void
}

; GCN-LABEL: {{^}}fold_mul_abs:
; GCN: load_dword [[V:v[0-9]+]]
; GCN: v_and_b32_e32 [[ABS:v[0-9]]], 0x7fffffff, [[V]]
; GCN: store_dword [[ABS]]

define amdgpu_kernel void @fold_mul_abs(ptr addrspace(1) %arg) {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, ptr addrspace(1) %arg, i32 %tid
  %v = load float, ptr addrspace(1) %gep, align 4
  %cmp = fcmp fast olt float %v, 0.000000e+00
  %sel = select i1 %cmp, float -1.000000e+00, float 1.000000e+00
  %mul = fmul fast float %v, %sel
  store float %mul, ptr addrspace(1) %gep, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #0

attributes #0 = { nounwind readnone speculatable }
