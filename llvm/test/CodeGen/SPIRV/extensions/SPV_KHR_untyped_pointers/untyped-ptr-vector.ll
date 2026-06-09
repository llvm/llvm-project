; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - -filetype=obj | spirv-val %}

; Test untyped pointers with vector types.

; CHECK: OpCapability UntypedPointersKHR
; CHECK: OpExtension "SPV_KHR_untyped_pointers"

; CHECK-DAG: OpTypeUntypedPointerKHR CrossWorkgroup

; CHECK: OpFunction
; CHECK: OpFunctionParameter
; CHECK: OpLoad
; CHECK: OpStore
define spir_kernel void @test_v4i32(ptr addrspace(1) %in, ptr addrspace(1) %out) {
entry:
  %val = load <4 x i32>, ptr addrspace(1) %in, align 16
  store <4 x i32> %val, ptr addrspace(1) %out, align 16
  ret void
}

; CHECK: OpFunction
; CHECK: OpFunctionParameter
; CHECK: OpLoad
; CHECK: OpStore
define spir_kernel void @test_v4f32(ptr addrspace(1) %in, ptr addrspace(1) %out) {
entry:
  %val = load <4 x float>, ptr addrspace(1) %in, align 16
  store <4 x float> %val, ptr addrspace(1) %out, align 16
  ret void
}

; CHECK: OpFunction
; CHECK: OpUntypedPtrAccessChainKHR
define spir_kernel void @test_vector_gep(ptr addrspace(1) %base, ptr addrspace(1) %out) {
entry:
  %ptr = getelementptr <4 x i32>, ptr addrspace(1) %base, i64 2
  %val = load <4 x i32>, ptr addrspace(1) %ptr, align 16
  store <4 x i32> %val, ptr addrspace(1) %out, align 16
  ret void
}

; CHECK: OpFunction
; CHECK: OpUntypedVariableKHR
define spir_kernel void @test_vector_alloca(ptr addrspace(1) %out) {
entry:
  %local = alloca <4 x float>, align 16
  store <4 x float> <float 1.0, float 2.0, float 3.0, float 4.0>, ptr %local, align 16
  %val = load <4 x float>, ptr %local, align 16
  store <4 x float> %val, ptr addrspace(1) %out, align 16
  ret void
}
