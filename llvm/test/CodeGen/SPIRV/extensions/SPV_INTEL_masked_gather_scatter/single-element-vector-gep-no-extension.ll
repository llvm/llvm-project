; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; A <1 x ptr> GEP scalarizes to a single spv_gep plus an
; insertelement/extractelement pair that folds away, so no vector-of-pointers
; SPIR-V type is materialized and SPV_INTEL_masked_gather_scatter is not
; required.

; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#PTR_I32:]] = OpTypePointer CrossWorkgroup %[[#I32]]
; CHECK-NOT: OpTypeVector %[[#PTR_I32]]

; CHECK: OpFunction
; CHECK: OpPtrAccessChain
; CHECK-NOT: OpCompositeInsert
; CHECK: OpLoad %[[#I32]]
; CHECK: OpFunctionEnd
define spir_kernel void @test_vector_gep_v1(ptr addrspace(1) %p, ptr addrspace(1) %out) {
  %gep = getelementptr i32, ptr addrspace(1) %p, <1 x i64> <i64 5>
  %elem = extractelement <1 x ptr addrspace(1)> %gep, i32 0
  %val = load i32, ptr addrspace(1) %elem
  store i32 %val, ptr addrspace(1) %out
  ret void
}
