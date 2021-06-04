; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -amdgpu-propagate-attributes-late  %s | FileCheck %s

; Test attributes on a function which
; is called indirectly from two kernels
; having different launch bounds.

; This function should not have any attributes on it.
; CHECK-LABEL: define float @foo() {
define float @foo() {
   ret float 0.0
}

define amdgpu_kernel void @kernel1(float *%result, i32 %type) #1 {
  %fn = alloca float ()*
  store float ()* @foo, float ()** %fn
  %fp = load float ()*, float ()** %fn
  %indirect_call = call contract float %fp()
  store float %indirect_call, float* %result
  ret void
}

define amdgpu_kernel void @kernel2(float *%result, i32 %type) #2 {
  %fn = alloca float ()*
  store float ()* @foo, float ()** %fn
  %fp = load float ()*, float ()** %fn
  %indirect_call = call contract float %fp()
  store float %indirect_call, float* %result
  ret void
}

attributes #1 = { "amdgpu-flat-work-group-size"="1,256" }
attributes #2 = { "amdgpu-flat-work-group-size"="1,512" }
