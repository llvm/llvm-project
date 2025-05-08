; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 -verify-machineinstrs %s -o - | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 -global-isel -verify-machineinstrs %s -o - | FileCheck %s

; CHECK: .cluster_dims:
; CHECK-NEXT: - 2
; CHECK-NEXT: - 2
; CHECK-NEXT: - 2
define dso_local amdgpu_kernel void @_Z15test_literal_3dv() #0 {
entry:
  ret void
}

; CHECK: .cluster_dims:
; CHECK-NEXT: - 2
; CHECK-NEXT: - 2
; CHECK-NEXT: - 1
define dso_local amdgpu_kernel void @_Z15test_literal_2dv() #1 {
entry:
  ret void
}

; CHECK: .cluster_dims:
; CHECK-NEXT: - 4
; CHECK-NEXT: - 1
; CHECK-NEXT: - 1
define dso_local amdgpu_kernel void @_Z15test_literal_1dv() #2 {
entry:
  ret void
}

; CHECK: .cluster_dims:
; CHECK-NEXT: - 4
; CHECK-NEXT: - 2
; CHECK-NEXT: - 1
define dso_local amdgpu_kernel void @_Z13test_constantv() #3 {
entry:
  ret void
}

attributes #0 = { convergent mustprogress noinline norecurse nounwind "amdgpu-cluster-dims"="2,2,2" }
attributes #1 = { convergent mustprogress noinline norecurse nounwind "amdgpu-cluster-dims"="2,2,1" }
attributes #2 = { convergent mustprogress noinline norecurse nounwind "amdgpu-cluster-dims"="4,1,1" }
attributes #3 = { convergent mustprogress noinline norecurse nounwind "amdgpu-cluster-dims"="4,2,1" }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"amdhsa_code_object_version", i32 600}
