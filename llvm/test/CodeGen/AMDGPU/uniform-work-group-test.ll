; RUN: opt -S -mtriple=amdgcn-amd- -amdgpu-annotate-kernel-features %s | FileCheck %s 

; CHECK: define void @func1() #[[FUNC:[0-9]+]] {
define void @func1() #0 {
  ret void
}

; CHECK: define void @func4() #[[FUNC]] {
define void @func4() #1 {
  ret void
}

; CHECK: define void @func2() #[[FUNC]] {
define void @func2() #1 {
  call void @func4()
  call void @func1()
  ret void
}

; CHECK: define void @func3() #[[FUNC]] {
define void @func3() #1 {
  call void @func1()
  ret void
}

; CHECK: define amdgpu_kernel void @kernel3() #[[FUNC]] {
define amdgpu_kernel void @kernel3() #2 {
  call void @func2()
  call void @func3()
  ret void
}

attributes #2 = { "uniform-work-group-size"="true" }

; CHECK: attributes #[[FUNC]] = { "uniform-work-group-size"="true" }
