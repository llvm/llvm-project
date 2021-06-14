; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -amdgpu-propagate-attributes-late  %s | FileCheck %s

; Test to check if we skip attributes on address
; taken functions and its callees.

; CHECK-LABEL: define float @bar() {
define float @bar() {
  ret float 0.0
}

; CHECK-LABEL: define float @baz() {
define float @baz() {
  ret float 0.0
}

; CHECK-LABEL: define float @foo() {
define float @foo() {
  %v1 = call contract float @bar()
  %v2 = call contract float @baz()
  %v3 = fadd float %v1, %v2
  ret float %v3
}

; CHECK-LABEL: define amdgpu_kernel void @kernel(float* %result, i32 %type) #0 {
define amdgpu_kernel void @kernel(float *%result, i32 %type) #1 {
  %fn = alloca float ()*
  store float ()* @foo, float ()** %fn
  %fp = load float ()*, float ()** %fn
  %indirect_call = call contract float %fp()
  store float %indirect_call, float* %result
  ret void
}

attributes #1 = { "amdgpu-flat-work-group-size"="1,256" }
