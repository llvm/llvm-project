; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -amdgpu-propagate-attributes-early  %s | FileCheck %s

; Complicated call graph where a function is called
; directly from a kernel abd also from a function 
; whose address is taken. 

; CHECK-LABEL: define float @common_callee.gc() #0 {
define float @common_callee.gc() {
   ret float 0.0
}

; CHECK-LABEL: define float @foo() {
define float @foo() {
   ret float 0.0
}

; CHECK-LABEL: define float @bar() {
define float @bar() {
   ret float 0.0
}

; CHECK-LABEL: define float @baz() {
define float @baz() {
   ret float 0.0
}

define amdgpu_kernel void @switch_indirect_kernel(float *%result, i32 %type) #1 {
  %fn = alloca float ()*
  switch i32 %type, label %sw.default [
    i32 1, label %sw.bb
    i32 2, label %sw.bb2
    i32 3, label %sw.bb3
  ]

sw.bb:
  store float ()* @foo, float ()** %fn
  br label %sw.epilog

sw.bb2:
  store float ()* @bar, float ()** %fn
  br label %sw.epilog

sw.bb3:
  store float ()* @baz, float ()** %fn
  br label %sw.epilog

sw.default:
  br label %sw.epilog

sw.epilog:
  %fp = load float ()*, float ()** %fn
  %direct_call = call contract float @common_callee.gc()
  %indirect_call = call contract float %fp()
  store float %indirect_call, float* %result
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,256" "target-features"="-wavefrontsize16,-wavefrontsize32,+wavefrontsize64" }
attributes #1 = { "amdgpu-flat-work-group-size"="1,256" } 
