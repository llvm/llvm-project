; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - -filetype=obj | spirv-val %}

; Arrays and structs nested into each other. One access chain carries the
; outermost aggregate as its Base Type plus every index of the walk.

%struct.Inner = type { i32, [4 x float] }
%struct.Outer = type { [3 x %struct.Inner], i32 }

; CHECK: OpCapability UntypedPointersKHR
; CHECK: OpExtension "SPV_KHR_untyped_pointers"

; CHECK-DAG: %[[#CROSS_PTR:]] = OpTypeUntypedPointerKHR CrossWorkgroup
; CHECK-DAG: %[[#F32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#I64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#LEN4:]] = OpConstant %[[#I32]] 4
; CHECK-DAG: %[[#FARR:]] = OpTypeArray %[[#F32]] %[[#LEN4]]
; CHECK-DAG: %[[#INNER:]] = OpTypeStruct %[[#I32]] %[[#FARR]]
; CHECK-DAG: %[[#LEN3:]] = OpConstant %[[#I32]] 3
; CHECK-DAG: %[[#INNERARR:]] = OpTypeArray %[[#INNER]] %[[#LEN3]]
; CHECK-DAG: %[[#OUTER:]] = OpTypeStruct %[[#INNERARR]] %[[#I32]]
; CHECK-DAG: %[[#CONST1_64:]] = OpConstant %[[#I64]] 1
; CHECK-DAG: %[[#CONST2_64:]] = OpConstant %[[#I64]] 2
; CHECK-DAG: %[[#CONST3_64:]] = OpConstant %[[#I64]] 3
; CHECK-DAG: %[[#CONST1_32:]] = OpConstant %[[#I32]] 1
; CHECK-DAG: %[[#NULL32:]] = OpConstantNull %[[#I32]]
; CHECK-DAG: %[[#NULL64:]] = OpConstantNull %[[#I64]]

; Walk p[1].inner[2].farr[3] in one chain.
; CHECK: OpFunction
; CHECK: %[[#P:]] = OpFunctionParameter %[[#CROSS_PTR]]
; CHECK: %[[#]] = OpUntypedPtrAccessChainKHR %[[#CROSS_PTR]] %[[#OUTER]] %[[#P]] %[[#CONST1_64]] %[[#NULL32]] %[[#CONST2_64]] %[[#CONST1_32]] %[[#CONST3_64]]
define spir_kernel void @nested_array_in_struct(ptr addrspace(1) %p, ptr addrspace(1) %out) {
entry:
  %q = getelementptr %struct.Outer, ptr addrspace(1) %p, i64 1, i32 0, i64 2, i32 1, i64 3
  %v = load float, ptr addrspace(1) %q, align 4
  store float %v, ptr addrspace(1) %out, align 4
  ret void
}

; An array of structs indexed by a runtime value, then a struct field.
; CHECK: OpFunction
; CHECK: %[[#P:]] = OpFunctionParameter %[[#CROSS_PTR]]
; CHECK: %[[#IDX:]] = OpFunctionParameter %[[#I64]]
; CHECK: %[[#]] = OpUntypedPtrAccessChainKHR %[[#CROSS_PTR]] %[[#INNERARR]] %[[#P]] %[[#NULL64]] %[[#IDX]] %[[#NULL32]]
define spir_kernel void @array_of_structs(ptr addrspace(1) %p, i64 %i, ptr addrspace(1) %out) {
entry:
  %q = getelementptr [3 x %struct.Inner], ptr addrspace(1) %p, i64 0, i64 %i, i32 0
  %v = load i32, ptr addrspace(1) %q, align 4
  store i32 %v, ptr addrspace(1) %out, align 4
  ret void
}
