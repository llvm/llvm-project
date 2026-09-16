; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Test that addrspacecast(null) is simplified to null of the target address
; space type in all contexts: composite constants, instruction operands, and
; global initializers.

%struct = type { i32, ptr addrspace(4) }

@gv = internal addrspace(1) constant %struct { i32 42, ptr addrspace(4) addrspacecast (ptr null to ptr addrspace(4)) }, align 8

; CHECK-DAG: %[[#INT:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#CHAR:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#GENPTR:]] = OpTypePointer Generic %[[#CHAR]]
; CHECK-DAG: %[[#CWGPTR:]] = OpTypePointer CrossWorkgroup %[[#CHAR]]
; CHECK-DAG: %[[#STRUCT:]] = OpTypeStruct %[[#INT]] %[[#GENPTR]]
; CHECK-DAG: %[[#CONST:]] = OpConstant %[[#INT]] 42
; CHECK-DAG: %[[#NULL:]] = OpConstantNull %[[#GENPTR]]
; CHECK-DAG: %[[#CWGNULL:]] = OpConstantNull %[[#CWGPTR]]
; CHECK-DAG: %[[#BOOL:]] = OpTypeBool

; Composite constant: addrspacecast(null) in struct member should become
; OpConstantNull of the pointer type.
; CHECK-DAG: OpConstantComposite %[[#STRUCT]] %[[#CONST]] %[[#NULL]]

define spir_kernel void @test(ptr addrspace(1) %out) {
entry:
  ret void
}

; Instruction operand: addrspacecast(null) in icmp should become
; OpConstantNull of the pointer type, not a cast from integer-typed null.
; CHECK: %[[#]] = OpPtrEqual %[[#BOOL]] %[[#CWGNULL]] %[[#]]
define spir_kernel void @test_icmp(ptr addrspace(1) %arg, ptr addrspace(1) %out) {
entry:
  %cmp = icmp eq ptr addrspace(1) addrspacecast (ptr null to ptr addrspace(1)), %arg
  %zext = zext i1 %cmp to i32
  store i32 %zext, ptr addrspace(1) %out
  ret void
}
