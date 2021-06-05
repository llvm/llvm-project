; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -amdgpu-propagate-attributes-late  %s | FileCheck %s

; Test to check if we skip propgating attributes even if
; a function is called directly as well as
; indirectly. "baz" is called directly as well indirectly.

; CHECK-LABEL: define float @foo()
define float @foo() #1 {
  ret float 0.0
}

; CHECK-LABEL: define float @bar()
define float @bar() #1 {
 ret float 0.0
}

; CHECK-LABEL: define float @baz()
define float @baz() #1 {
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
  %direct_call = call contract float @baz()
  %indirect_call = call contract float %fp()
  store float %indirect_call, float* %result
  ret void
}

attributes #1 = { "amdgpu-flat-work-group-size"="1,256" }
