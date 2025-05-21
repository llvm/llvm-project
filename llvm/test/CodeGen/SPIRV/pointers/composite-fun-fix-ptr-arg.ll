; The goal of the test case is to ensure that if module contains functions with mutated signature
; (due to preprocessing of aggregate types), functions still are going through re-creating of
; function type to preserve pointee type information for arguments.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#Int8:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#Half:]] = OpTypeFloat 16
; CHECK-DAG: %[[#Struct:]] = OpTypeStruct %[[#Half]]
; CHECK-DAG: %[[#Void:]] = OpTypeVoid
; CHECK-DAG: %[[#PtrInt8:]] = OpTypePointer CrossWorkgroup %[[#Int8:]]
; CHECK-DAG: %[[#FooType:]] = OpTypeFunction %[[#Void]] %[[#PtrInt8]] %[[#Struct]]
; CHECK-DAG: %[[#Int64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#PtrInt64:]] = OpTypePointer CrossWorkgroup %[[#Int64]]
; CHECK-DAG: %[[#BarType:]] = OpTypeFunction %[[#Void]] %[[#PtrInt64]] %[[#Struct]]
; CHECK: OpFunction %[[#Void]] None %[[#FooType]]
; CHECK: OpFunctionParameter %[[#PtrInt8]]
; CHECK: OpFunctionParameter %[[#Struct]]
; CHECK: OpFunction %[[#Void]] None %[[#BarType]]
; CHECK: OpFunctionParameter %[[#PtrInt64]]
; CHECK: OpFunctionParameter %[[#Struct]]

%t_half = type { half }

define spir_kernel void @foo(ptr addrspace(1) %a, %t_half %b) {
entry:
  ret void
}


define spir_kernel void @bar(ptr addrspace(1) %a, %t_half %b) {
entry:
  %r = getelementptr inbounds i64, ptr addrspace(1) %a, i64 0
  ret void
}

declare spir_func %t_half @_Z29__spirv_SpecConstantComposite(half)
