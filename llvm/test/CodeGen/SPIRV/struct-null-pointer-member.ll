; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Test that a null pointer member in a struct constant (via addrspacecast)
; gets the correct pointer type in OpConstantComposite, not integer type.

%struct = type { i32, ptr addrspace(4) }

@gv = internal addrspace(1) constant %struct { i32 42, ptr addrspace(4) addrspacecast (ptr null to ptr addrspace(4)) }, align 8

; CHECK-DAG: %[[#INT:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#CHAR:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#GENPTR:]] = OpTypePointer Generic %[[#CHAR]]
; CHECK-DAG: %[[#STRUCT:]] = OpTypeStruct %[[#INT]] %[[#GENPTR]]
; CHECK-DAG: %[[#CONST:]] = OpConstant %[[#INT]] 42
; CHECK-DAG: %[[#NULL:]] = OpConstantNull %[[#GENPTR]]
; CHECK: OpConstantComposite %[[#STRUCT]] %[[#CONST]] %[[#NULL]]

define spir_kernel void @test(ptr addrspace(1) %out) {
entry:
  ret void
}
