; RUN: opt -S -mtriple=amdgcn-amd- -amdgpu-annotate-kernel-features %s | FileCheck %s

; If the kernel does not have the uniform-work-group-attribute, set both callee and caller as false

; CHECK: define void @foo() #[[FOO:[0-9]+]] {
define void @foo() #0 {
  ret void
}

; CHECK: define amdgpu_kernel void @kernel1() #[[KERNEL1:[0-9]+]] {
define amdgpu_kernel void @kernel1() #1 {
  call void @foo()
  ret void
}

attributes #0 = { "uniform-work-group-size"="true" }

; CHECK: attributes #[[FOO]] = { "uniform-work-group-size"="false" }
; CHECK: attributes #[[KERNEL1]] = { "amdgpu-calls" "uniform-work-group-size"="false" }
