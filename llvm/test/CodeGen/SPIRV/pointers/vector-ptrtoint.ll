; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

declare spir_func void @foo(<2 x i64>)

; CHECK-DAG: %[[#I64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#V2I64:]] = OpTypeVector %[[#I64]] 2
; CHECK-DAG: OpName %[[#FUNC:]] "test_scalar_ptrtoint_to_vector"

; CHECK: %[[#FUNC]] = OpFunction
; CHECK: OpConvertPtrToU %[[#I64]]
; CHECK: OpConvertPtrToU %[[#I64]]
; CHECK: OpCompositeInsert %[[#V2I64]]
; CHECK: OpCompositeInsert %[[#V2I64]]
; CHECK: OpFunctionEnd

define spir_kernel void @test_scalar_ptrtoint_to_vector(ptr addrspace(1) %p0, ptr addrspace(1) %p1) {
entry:
  ; Convert each pointer to integer separately
  %addr0 = ptrtoint ptr addrspace(1) %p0 to i64
  %addr1 = ptrtoint ptr addrspace(1) %p1 to i64
  ; Combine into a vector
  %vec0 = insertelement <2 x i64> poison, i64 %addr0, i32 0
  %vec = insertelement <2 x i64> %vec0, i64 %addr1, i32 1
  call void @foo(<2 x i64> %vec)
  ret void
}
