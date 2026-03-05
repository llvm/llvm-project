; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

declare spir_func void @foo(<2 x i64>)

; CHECK-DAG: %[[#I64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#I8:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#PTR:]] = OpTypePointer CrossWorkgroup %[[#I8]]
; CHECK-DAG: %[[#V2I64:]] = OpTypeVector %[[#I64]] 2
; CHECK-DAG: OpName %[[#P0:]] "p.elem0"
; CHECK-DAG: OpName %[[#P1:]] "p.elem1"

; CHECK: %[[#ENTRY:]] = OpFunction
; CHECK: %[[#P0]] = OpFunctionParameter %[[#PTR]]
; CHECK: %[[#P1]] = OpFunctionParameter %[[#PTR]]
; CHECK: OpFunctionParameter %[[#PTR]]
; CHECK: OpFunctionParameter %[[#PTR]]

; CHECK: %[[#CONV0:]] = OpConvertPtrToU %[[#I64]] %[[#P0]]
; CHECK: OpCompositeInsert %[[#V2I64]] %[[#CONV0]]
; CHECK: %[[#CONV1:]] = OpConvertPtrToU %[[#I64]] %[[#P1]]
; CHECK: OpCompositeInsert %[[#V2I64]] %[[#CONV1]]

define spir_kernel void @test_ptrtoaddr(<2 x ptr addrspace(1)> %p, <2 x ptr addrspace(1)> %res) {
entry:
  %addr = ptrtoint <2 x ptr addrspace(1)> %p to <2 x i64>
  call void @foo(<2 x i64> %addr)
  ret void
}
