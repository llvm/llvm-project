; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -amdgpu-annotate-kernel-features %s | FileCheck %s

; CHECK: define void @func1() #[[FUNC:[0-9]+]] {
define void @func1() {
  ret void
}

; CHECK: define void @func4() #[[FUNC]] {
define void @func4() {
  ret void
}

; CHECK: define void @func2() #[[FUNC]] {
define void @func2() #0 {
  call void @func4()
  call void @func1()
  ret void
}

; CHECK: define void @func3() #[[FUNC]] {
define void @func3() {
  call void @func1()
  ret void
}

; CHECK: define amdgpu_kernel void @kernel3() #[[FUNC:[0-9]+]] {
define amdgpu_kernel void @kernel3() #0 {
  call void @func2()
  call void @func3()
  ret void
}

attributes #0 = { "uniform-work-group-size"="false" }

; CHECK: attributes #[[FUNC]] = { "amdgpu-calls" "uniform-work-group-size"="false" }
