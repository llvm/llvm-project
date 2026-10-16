; RUN: llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - | FileCheck %s --implicit-check-not=MaxByteOffset --implicit-check-not="OpCapability Addresses"
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; MaxByteOffset requires Addresses, which is not available for shader targets.
; CHECK: OpCapability Shader
; CHECK: OpName %[[#PTR:]] "ptr"
; CHECK: %[[#INT:]] = OpTypeInt 32 0
; CHECK: %[[#PTR_T:]] = OpTypePointer Function %[[#INT]]
; CHECK: %[[#PTR]] = OpFunctionParameter %[[#PTR_T]]

define internal void @helper(ptr dereferenceable(4) %ptr) noinline {
  store i32 1, ptr %ptr, align 4
  ret void
}

define void @main() #0 {
  %ptr = alloca i32, align 4
  call void @helper(ptr %ptr)
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
