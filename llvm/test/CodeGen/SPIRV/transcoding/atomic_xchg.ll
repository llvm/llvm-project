; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#INT:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#LONG:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#PTR:]] = OpTypePointer CrossWorkgroup %[[#LONG]]
; CHECK-DAG: %[[#PARAM:]] = OpFunctionParameter %[[#PTR]]
; CHECK-DAG: %[[#VALUE:]] = OpFunctionParameter %[[#LONG]]
; CHECK-DAG: %[[#CONST0:]] = OpConstant %[[#INT]] 0
; CHECK-DAG: %[[#CONST2:]] = OpConstant %[[#INT]] 2
; CHECK: %[[#]] = OpAtomicExchange %[[#LONG]] %[[#PARAM]] %[[#CONST2]] %[[#CONST0]] %[[#VALUE]]
define void @test_atomic_xchg(ptr addrspace(1) %p, i64 %val) {
  %call1 = call i64 @_Z9atom_xchgPU3AS1Vll(ptr addrspace(1) %p, i64 %val)
  ret void
}

declare i64 @_Z9atom_xchgPU3AS1Vll(ptr addrspace(1), i64)
