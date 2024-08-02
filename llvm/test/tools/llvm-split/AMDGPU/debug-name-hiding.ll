; RUN: llvm-split -o %t %s -j 3 -mtriple amdgcn-amd-amdhsa -debug -amdgpu-module-splitting-log-private 2>&1 | FileCheck %s --implicit-check-not=MyCustomKernel
; REQUIRES: asserts

; SHA256 of the kernel names.

; CHECK: a097723d21cf9f35d90e6fb7881995ac8c398b3366a6c97efc657404f9fe301c
; CHECK: 626bc23242de8fcfda7f0e66318d29455c081df6b5380e64d14703c95fcbcd59
; CHECK: c38d90a7ca71dc5d694bb9e093dadcdedfc4cb4adf7ed7e46d42fe95a0b4ef55

define amdgpu_kernel void @MyCustomKernel0() {
  ret void
}

define amdgpu_kernel void @MyCustomKernel1() {
  ret void
}

define amdgpu_kernel void @MyCustomKernel2() {
  ret void
}
