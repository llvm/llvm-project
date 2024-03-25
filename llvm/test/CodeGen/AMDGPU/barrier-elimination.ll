; RUN: llc -mtriple=amdgcn < %s | FileCheck %s
; RUN: llc -mtriple=amdgcn < %s -global-isel | FileCheck %s

; CHECK-LABEL: {{^}}unknown_wgs:
; CHECK: s_barrier
define amdgpu_kernel void @unknown_wgs() {
  tail call void @llvm.amdgcn.s.barrier() #0
  ret void
}

; CHECK-LABEL: {{^}}flat_wgs_attr_32_128:
; CHECK: s_barrier
define amdgpu_kernel void @flat_wgs_attr_32_128() #1 {
  tail call void @llvm.amdgcn.s.barrier() #0
  ret void
}

; CHECK-LABEL: {{^}}flat_wgs_attr_32_64:
; CHECK: :
; CHECK-NEXT: ; wave barrier
; CHECK-NEXT: s_endpgm
define amdgpu_kernel void @flat_wgs_attr_32_64() #2 {
  tail call void @llvm.amdgcn.s.barrier() #0
  ret void
}

declare void @llvm.amdgcn.s.barrier() #0

attributes #0 = { convergent nounwind }
attributes #1 = { nounwind "amdgpu-flat-work-group-size"="32,128" }
attributes #2 = { nounwind "amdgpu-flat-work-group-size"="32,64" }
