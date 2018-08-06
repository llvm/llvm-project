; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=fiji  -memdep-block-scan-limit=1 -amdgpu-scalarize-global-loads -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN %s

; GCN-LABEL: {{^}}unknown_memdep_analysis:
; GCN: flat_load_dword
; GCN: flat_load_dword
; GCN: flat_store_dword
define amdgpu_kernel void @unknown_memdep_analysis(float addrspace(1)* nocapture readonly %arg, float %arg1) #0 {
bb:
  %tmp53 = load float, float addrspace(1)* undef, align 4
  %tmp54 = getelementptr inbounds float, float addrspace(1)* %arg, i32 31
  %tmp55 = load float, float addrspace(1)* %tmp54, align 4
  %tmp56 = tail call float @llvm.fmuladd.f32(float %arg1, float %tmp53, float %tmp55)
  store float %tmp56, float addrspace(1)* undef, align 4
  ret void
}

declare float @llvm.fmuladd.f32(float, float, float) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone speculatable }
