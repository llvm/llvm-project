; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -amdgpu-propagate-attributes-late %s | FileCheck %s

; CHECK: define internal void @max_flat_1_1024() #0 {
define internal void @max_flat_1_1024() #0 {
  ret void
}

; CHECK: define internal void @max_flat_1_256() #1 {
define internal void @max_flat_1_256() #1 {
  ret void
}

; CHECK: define amdgpu_kernel void @kernel_1_256_call_default() #1 {
define amdgpu_kernel void @kernel_1_256_call_default() #1 {
  call void @default()
  ret void
}

; CHECK: define amdgpu_kernel void @kernel_1_256_call_1_256() #1 {
define amdgpu_kernel void @kernel_1_256_call_1_256() #1 {
  call void @max_flat_1_256()
  ret void
}

; CHECK: define amdgpu_kernel void @kernel_1_256_call_64_64() #1 {
define amdgpu_kernel void @kernel_1_256_call_64_64() #1 {
  call void @max_flat_64_64()
  ret void
}

; CHECK: define internal void @max_flat_64_64() #2 {
define internal void @max_flat_64_64() #2 {
  ret void
}

; CHECK: define internal void @default() #2 {
define internal void @default() #3 {
  ret void
}

attributes #0 = { noinline "amdgpu-flat-work-group-size"="1,1024" }
attributes #1 = { noinline "amdgpu-flat-work-group-size"="1,256" }
attributes #2 = { noinline "amdgpu-flat-work-group-size"="64,64" }
attributes #3 = { noinline }

; CHECK: attributes #0 = { noinline "amdgpu-flat-work-group-size"="1,1024"
; CHECK-NEXT: attributes #1 = { noinline "amdgpu-flat-work-group-size"="1,256"
; CHECK-NEXT: attributes #2 = { noinline "amdgpu-flat-work-group-size"="1,256"
