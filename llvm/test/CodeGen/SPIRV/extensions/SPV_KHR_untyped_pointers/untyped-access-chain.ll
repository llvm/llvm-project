; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - -filetype=obj | spirv-val %}

; Test OpUntypedPtrAccessChainKHR for GEP instructions in physical SPIR-V.

; CHECK: OpCapability UntypedPointersKHR
; CHECK: OpExtension "SPV_KHR_untyped_pointers"

; CHECK: OpUntypedPtrAccessChainKHR
define spir_kernel void @test_gep_const_index(ptr addrspace(1) %base, ptr addrspace(1) %out) {
entry:
  %ptr = getelementptr i32, ptr addrspace(1) %base, i64 5
  %val = load i32, ptr addrspace(1) %ptr, align 4
  store i32 %val, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK: OpUntypedPtrAccessChainKHR
define spir_kernel void @test_gep_var_index(ptr addrspace(1) %base, i64 %idx, ptr addrspace(1) %out) {
entry:
  %ptr = getelementptr i32, ptr addrspace(1) %base, i64 %idx
  %val = load i32, ptr addrspace(1) %ptr, align 4
  store i32 %val, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK: OpUntypedInBoundsPtrAccessChainKHR
define spir_kernel void @test_gep_inbounds(ptr addrspace(1) %base, ptr addrspace(1) %out) {
entry:
  %ptr = getelementptr inbounds i32, ptr addrspace(1) %base, i64 3
  %val = load i32, ptr addrspace(1) %ptr, align 4
  store i32 %val, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK: OpFunction
; CHECK: OpUntypedPtrAccessChainKHR
define spir_kernel void @test_gep_i8(ptr addrspace(1) %base, ptr addrspace(1) %out) {
entry:
  %ptr = getelementptr i8, ptr addrspace(1) %base, i64 10
  %val = load i8, ptr addrspace(1) %ptr, align 1
  store i8 %val, ptr addrspace(1) %out, align 1
  ret void
}

; CHECK: OpFunction
; CHECK: OpUntypedPtrAccessChainKHR
define spir_kernel void @test_gep_i64(ptr addrspace(1) %base, ptr addrspace(1) %out) {
entry:
  %ptr = getelementptr i64, ptr addrspace(1) %base, i64 2
  %val = load i64, ptr addrspace(1) %ptr, align 8
  store i64 %val, ptr addrspace(1) %out, align 8
  ret void
}

; CHECK: OpFunction
; CHECK: OpUntypedPtrAccessChainKHR
define spir_kernel void @test_gep_float(ptr addrspace(1) %base, ptr addrspace(1) %out) {
entry:
  %ptr = getelementptr float, ptr addrspace(1) %base, i64 4
  %val = load float, ptr addrspace(1) %ptr, align 4
  store float %val, ptr addrspace(1) %out, align 4
  ret void
}
