; RUN: opt -S -mtriple=amdgcn-amd- -amdgpu-annotate-kernel-features %s | FileCheck %s

; Propagate the uniform-work-group-attribute from the kernel to callee if it doesn't have it
; CHECK: define void @func() #[[FUNC:[0-9]+]] {
define void @func() #0 {
  ret void
}

; CHECK: define amdgpu_kernel void @kernel1() #[[KERNEL1:[0-9]+]] {
define amdgpu_kernel void @kernel1() #1 {
  call void @func()
  ret void
}

; External declaration of a function
; CHECK: define weak_odr void @weak_func() #[[FUNC]] {
define weak_odr void @weak_func() #0 {
  ret void
}

; CHECK: define amdgpu_kernel void @kernel2() #[[KERNEL2:[0-9]+]] {
define amdgpu_kernel void @kernel2() #2 {
  call void @weak_func()
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { "uniform-work-group-size"="false" }
attributes #2 = { "uniform-work-group-size"="true" }

; CHECK: attributes #[[FUNC]] = { nounwind "uniform-work-group-size"="false" }
; CHECK: attributes #[[KERNEL1]] = { "amdgpu-calls" "uniform-work-group-size"="false" }
; CHECK: attributes #[[KERNEL2]] = { "amdgpu-calls" "uniform-work-group-size"="true" }
