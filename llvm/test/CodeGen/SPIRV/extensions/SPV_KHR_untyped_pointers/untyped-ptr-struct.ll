; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - -filetype=obj | spirv-val %}

; Test untyped pointers with struct types.

; CHECK: OpCapability UntypedPointersKHR
; CHECK: OpExtension "SPV_KHR_untyped_pointers"

; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#F32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#CROSS_PTR:]] = OpTypeUntypedPointerKHR CrossWorkgroup

%struct.Point = type { float, float }
%struct.Data = type { i32, float, i32 }

; CHECK: OpFunction
; CHECK: OpFunctionParameter %[[#CROSS_PTR]]
define spir_kernel void @test_struct_load(ptr addrspace(1) %point, ptr addrspace(1) %out) {
entry:
  %x = load float, ptr addrspace(1) %point, align 4
  store float %x, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK: OpFunction
; CHECK: OpUntypedPtrAccessChainKHR %[[#CROSS_PTR]]
define spir_kernel void @test_struct_gep(ptr addrspace(1) %point, ptr addrspace(1) %out) {
entry:
  %y_ptr = getelementptr %struct.Point, ptr addrspace(1) %point, i64 0, i32 1
  %y = load float, ptr addrspace(1) %y_ptr, align 4
  store float %y, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK: OpFunction
; CHECK: OpUntypedPtrAccessChainKHR
; CHECK: OpUntypedPtrAccessChainKHR
define spir_kernel void @test_struct_multi_field(ptr addrspace(1) %data, ptr addrspace(1) %out_i, ptr addrspace(1) %out_f) {
entry:
  %field0_ptr = getelementptr %struct.Data, ptr addrspace(1) %data, i64 0, i32 0
  %field1_ptr = getelementptr %struct.Data, ptr addrspace(1) %data, i64 0, i32 1
  %val0 = load i32, ptr addrspace(1) %field0_ptr, align 4
  %val1 = load float, ptr addrspace(1) %field1_ptr, align 4
  store i32 %val0, ptr addrspace(1) %out_i, align 4
  store float %val1, ptr addrspace(1) %out_f, align 4
  ret void
}

; CHECK: OpFunction
; CHECK: OpUntypedVariableKHR
define spir_kernel void @test_struct_alloca(ptr addrspace(1) %out) {
entry:
  %local = alloca %struct.Point, align 4
  %x_ptr = getelementptr %struct.Point, ptr %local, i64 0, i32 0
  store float 1.0, ptr %x_ptr, align 4
  %x = load float, ptr %x_ptr, align 4
  store float %x, ptr addrspace(1) %out, align 4
  ret void
}
