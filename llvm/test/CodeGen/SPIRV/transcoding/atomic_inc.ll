; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#INT:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#LONG:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#PTR:]] = OpTypePointer CrossWorkgroup %[[#LONG]]
; CHECK-DAG: %[[#PARAM:]] = OpFunctionParameter %[[#PTR]]
; CHECK-DAG: %[[#CONST0:]] = OpConstant %[[#INT]] 0
; CHECK-DAG: %[[#CONST2:]] = OpConstant %[[#INT]] 2
; CHECK: %[[#]] = OpAtomicIIncrement %[[#LONG]] %[[#PARAM]] %[[#CONST2]] %[[#CONST0]]
define void @test_atomic_inc(ptr addrspace(1) %p, i64 %val) {
  %call1 = call i64 @_Z8atom_incPU3AS3Vl(ptr addrspace(1) %p)
  ret void
}

declare i64 @_Z8atom_incPU3AS3Vl(ptr addrspace(1))
