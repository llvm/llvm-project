; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - -filetype=obj | spirv-val %}

; Test untyped pointer types for different storage classes and data types.

; CHECK: OpCapability UntypedPointersKHR
; CHECK: OpExtension "SPV_KHR_untyped_pointers"

; CHECK-DAG: OpTypeUntypedPointerKHR CrossWorkgroup
; CHECK-DAG: OpTypeUntypedPointerKHR Workgroup

define spir_kernel void @test_i8_ptr(ptr addrspace(1) %in, ptr addrspace(1) %out) {
entry:
  %val = load i8, ptr addrspace(1) %in, align 1
  store i8 %val, ptr addrspace(1) %out, align 1
  ret void
}

define spir_kernel void @test_i16_ptr(ptr addrspace(1) %in, ptr addrspace(1) %out) {
entry:
  %val = load i16, ptr addrspace(1) %in, align 2
  store i16 %val, ptr addrspace(1) %out, align 2
  ret void
}

define spir_kernel void @test_i32_ptr(ptr addrspace(1) %in, ptr addrspace(1) %out) {
entry:
  %val = load i32, ptr addrspace(1) %in, align 4
  store i32 %val, ptr addrspace(1) %out, align 4
  ret void
}

define spir_kernel void @test_i64_ptr(ptr addrspace(1) %in, ptr addrspace(1) %out) {
entry:
  %val = load i64, ptr addrspace(1) %in, align 8
  store i64 %val, ptr addrspace(1) %out, align 8
  ret void
}

define spir_kernel void @test_f32_ptr(ptr addrspace(1) %in, ptr addrspace(1) %out) {
entry:
  %val = load float, ptr addrspace(1) %in, align 4
  store float %val, ptr addrspace(1) %out, align 4
  ret void
}

define spir_kernel void @test_f64_ptr(ptr addrspace(1) %in, ptr addrspace(1) %out) {
entry:
  %val = load double, ptr addrspace(1) %in, align 8
  store double %val, ptr addrspace(1) %out, align 8
  ret void
}

; CHECK: OpFunctionParameter
define spir_kernel void @test_local_ptr(ptr addrspace(3) %local_mem) {
entry:
  store i32 42, ptr addrspace(3) %local_mem, align 4
  ret void
}
