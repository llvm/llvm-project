; RUN: opt -S -mtriple=amdgcn-amd- -amdgpu-annotate-kernel-features %s | FileCheck %s

; Two kernels with different values of the uniform-work-group-attribute call the same function

; CHECK: define void @func() #[[FUNC:[0-9]+]] {
define void @func() #0 {
  ret void
}

; CHECK: define amdgpu_kernel void @kernel1() #[[KERNEL1:[0-9]+]] {
define amdgpu_kernel void @kernel1() #1 {
  call void @func()
  ret void
}

; CHECK: define amdgpu_kernel void @kernel2() #[[KERNEL2:[0-9]+]] {
define amdgpu_kernel void @kernel2() #2 {
  call void @func()
  ret void
}

attributes #1 = { "uniform-work-group-size"="true" }

; CHECK: attributes #[[FUNC]] = { "uniform-work-group-size"="false" }
; CHECK: attributes #[[KERNEL1]] = { "amdgpu-calls" "uniform-work-group-size"="true" }
; CHECK: attributes #[[KERNEL2]] = { "amdgpu-calls" "uniform-work-group-size"="false" }
