; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-amd-amdhsa %s -o - | FileCheck %s

; CHECK-LABEL: Begin function test_vector_gep_with_load
; CHECK: OpPtrAccessChain
; CHECK: OpLoad
; CHECK: OpStore
; CHECK: OpFunctionEnd
define spir_kernel void @test_vector_gep_with_load(ptr addrspace(1) %p, ptr addrspace(1) %out) {
  %gep = getelementptr i32, ptr addrspace(1) %p, <1 x i64> zeroinitializer
  %elem = extractelement <1 x ptr addrspace(1)> %gep, i32 0
  %val = load i32, ptr addrspace(1) %elem
  store i32 %val, ptr addrspace(1) %out
  ret void
}
