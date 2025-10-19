; Test that alloca with aggregate type generates correct OpVariable
; with the array type as the pointee, not a pointer-to-pointer type
;
; This test verifies that when we have an alloca of an array containing
; structs with pointers, the OpVariable uses the correct array type
; instead of incorrectly using a pointer-to-pointer type.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#Int8:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#Int64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#Int32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#One:]] = OpConstant %[[#Int32]] 1
; CHECK-DAG: %[[#Two:]] = OpConstant %[[#Int32]] 2
; CHECK-DAG: %[[#PtrCross:]] = OpTypePointer CrossWorkgroup %[[#Int8]]
; CHECK-DAG: %[[#Array1:]] = OpTypeArray %[[#Int64]] %[[#One]]
; CHECK-DAG: %[[#Struct1:]] = OpTypeStruct %[[#PtrCross]] %[[#Int64]] %[[#Array1]] %[[#Int64]]
; CHECK-DAG: %[[#Array2:]] = OpTypeArray %[[#Array1]] %[[#Two]]
; CHECK-DAG: %[[#Struct2:]] = OpTypeStruct %[[#Struct1]] %[[#Array2]]
; CHECK-DAG: %[[#Struct3:]] = OpTypeStruct %[[#Struct2]]
; CHECK-DAG: %[[#ArrayStruct:]] = OpTypeArray %[[#Struct3]] %[[#One]]
; CHECK-DAG: %[[#PtrFunc:]] = OpTypePointer Function %[[#ArrayStruct]]

; Verify OpVariable uses the array type, not pointer-to-pointer
; CHECK: %[[#Var:]] = OpVariable %[[#PtrFunc]] Function

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv64-unknown-unknown"

define spir_kernel void @test_alloca_aggregate() local_unnamed_addr {
entry:
  %y = alloca [1 x { { { ptr addrspace(1), i64, [1 x i64], i64 }, [2 x [1 x i64]] } }], align 8
  %ptr = load ptr addrspace(1), ptr %y, align 8
  ret void
}
