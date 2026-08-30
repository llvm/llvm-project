; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - -filetype=obj | spirv-val %}

; Test OpUntypedPtrAccessChainKHR for GEP in physical SPIR-V. The Base Type
; operand must be the GEP source element type, and the index must be unchanged.

; CHECK: OpCapability UntypedPointersKHR
; CHECK: OpExtension "SPV_KHR_untyped_pointers"

; CHECK-DAG: %[[#CROSS_PTR:]] = OpTypeUntypedPointerKHR CrossWorkgroup
; CHECK-DAG: %[[#I64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#I8:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#F32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#CONST5:]] = OpConstant %[[#I64]] 5
; CHECK-DAG: %[[#CONST3:]] = OpConstant %[[#I64]] 3
; CHECK-DAG: %[[#CONST10:]] = OpConstant %[[#I64]] 10
; CHECK-DAG: %[[#CONST2:]] = OpConstant %[[#I64]] 2
; CHECK-DAG: %[[#CONST4:]] = OpConstant %[[#I64]] 4

; CHECK: OpFunction
; CHECK: %[[#BASE:]] = OpFunctionParameter %[[#CROSS_PTR]]
; CHECK: %[[#]] = OpUntypedPtrAccessChainKHR %[[#CROSS_PTR]] %[[#I32]] %[[#BASE]] %[[#CONST5]]
define spir_kernel void @test_gep_const_index(ptr addrspace(1) %base, ptr addrspace(1) %out) {
entry:
  %ptr = getelementptr i32, ptr addrspace(1) %base, i64 5
  %val = load i32, ptr addrspace(1) %ptr, align 4
  store i32 %val, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK: OpFunction
; CHECK: %[[#BASE:]] = OpFunctionParameter %[[#CROSS_PTR]]
; CHECK: %[[#IDX:]] = OpFunctionParameter %[[#I64]]
; CHECK: %[[#]] = OpUntypedPtrAccessChainKHR %[[#CROSS_PTR]] %[[#I32]] %[[#BASE]] %[[#IDX]]
define spir_kernel void @test_gep_var_index(ptr addrspace(1) %base, i64 %idx, ptr addrspace(1) %out) {
entry:
  %ptr = getelementptr i32, ptr addrspace(1) %base, i64 %idx
  %val = load i32, ptr addrspace(1) %ptr, align 4
  store i32 %val, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK: OpFunction
; CHECK: %[[#BASE:]] = OpFunctionParameter %[[#CROSS_PTR]]
; CHECK: %[[#]] = OpUntypedInBoundsPtrAccessChainKHR %[[#CROSS_PTR]] %[[#I32]] %[[#BASE]] %[[#CONST3]]
define spir_kernel void @test_gep_inbounds(ptr addrspace(1) %base, ptr addrspace(1) %out) {
entry:
  %ptr = getelementptr inbounds i32, ptr addrspace(1) %base, i64 3
  %val = load i32, ptr addrspace(1) %ptr, align 4
  store i32 %val, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK: OpFunction
; CHECK: %[[#BASE:]] = OpFunctionParameter %[[#CROSS_PTR]]
; CHECK: %[[#]] = OpUntypedPtrAccessChainKHR %[[#CROSS_PTR]] %[[#I8]] %[[#BASE]] %[[#CONST10]]
define spir_kernel void @test_gep_i8(ptr addrspace(1) %base, ptr addrspace(1) %out) {
entry:
  %ptr = getelementptr i8, ptr addrspace(1) %base, i64 10
  %val = load i8, ptr addrspace(1) %ptr, align 1
  store i8 %val, ptr addrspace(1) %out, align 1
  ret void
}

; CHECK: OpFunction
; CHECK: %[[#BASE:]] = OpFunctionParameter %[[#CROSS_PTR]]
; CHECK: %[[#]] = OpUntypedPtrAccessChainKHR %[[#CROSS_PTR]] %[[#I64]] %[[#BASE]] %[[#CONST2]]
define spir_kernel void @test_gep_i64(ptr addrspace(1) %base, ptr addrspace(1) %out) {
entry:
  %ptr = getelementptr i64, ptr addrspace(1) %base, i64 2
  %val = load i64, ptr addrspace(1) %ptr, align 8
  store i64 %val, ptr addrspace(1) %out, align 8
  ret void
}

; CHECK: OpFunction
; CHECK: %[[#BASE:]] = OpFunctionParameter %[[#CROSS_PTR]]
; CHECK: %[[#]] = OpUntypedPtrAccessChainKHR %[[#CROSS_PTR]] %[[#F32]] %[[#BASE]] %[[#CONST4]]
define spir_kernel void @test_gep_float(ptr addrspace(1) %base, ptr addrspace(1) %out) {
entry:
  %ptr = getelementptr float, ptr addrspace(1) %base, i64 4
  %val = load float, ptr addrspace(1) %ptr, align 4
  store float %val, ptr addrspace(1) %out, align 4
  ret void
}
