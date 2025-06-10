; RUN: split-file %s %t

; RUN: not --crash llc -mtriple=amdgcn -mcpu=gfx1300 -filetype=null %t/spatial-cluster-1d-err.ll 2>&1 | FileCheck -check-prefix=CHECK-ERROR1 %s
; RUN: not --crash llc -mtriple=amdgcn -mcpu=gfx1300 -filetype=null %t/spatial-cluster-wg-err.ll 2>&1 | FileCheck -check-prefix=CHECK-ERROR2 %s
; RUN: not --crash llc -mtriple=amdgcn -mcpu=gfx1300 -filetype=null %t/spatial-cluster-fixed-dims-err.ll 2>&1 | FileCheck -check-prefix=CHECK-ERROR3 %s

;--- spatial-cluster-1d-err.ll
; CHECK-ERROR1: LLVM ERROR: Spatial cluster kernel is not 1D
define amdgpu_kernel void @non_1d_dims() #0 !reqd_work_group_size !{i32 32, i32 8, i32 1} {
entry:
  ret void
}
attributes #0 = { "amdgpu-cluster-dims"="2,2,1" "amdgpu-wavegroup-enable" "amdgpu-spatial-cluster" }

;--- spatial-cluster-wg-err.ll
; CHECK-ERROR2: LLVM ERROR: Spatial cluster kernel is not wavegroup kernel
define amdgpu_kernel void @non_wavegroup_kernel() #1 !reqd_work_group_size !{i32 32, i32 8, i32 1} {
entry:
  ret void
}
attributes #1 = { "amdgpu-cluster-dims"="2,1,1" "amdgpu-spatial-cluster" }

;--- spatial-cluster-fixed-dims-err.ll
; CHECK-ERROR3: LLVM ERROR: Spatial cluster kernel has non fixed cluster dims
define amdgpu_kernel void @non_fixed_dims() #2 !reqd_work_group_size !{i32 32, i32 8, i32 1} {
entry:
  ret void
}
attributes #2 = { "amdgpu-wavegroup-enable" "amdgpu-spatial-cluster" }
