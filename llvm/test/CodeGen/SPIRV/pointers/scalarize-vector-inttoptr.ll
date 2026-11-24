; Test that inttoptr on a vector of integers to a vector of pointers is
; scalarized into per-element inttoptr operations, since SPIR-V does not
; support vectors of pointers without extensions.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#I64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#I8:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#PTR:]] = OpTypePointer CrossWorkgroup %[[#I8]]

; CHECK-LABEL: Begin function test_inttoptr_scalarized
; CHECK-NEXT: %[[#A:]] = OpFunctionParameter
; CHECK: %[[#E0:]] = OpCompositeExtract %[[#I64]] %[[#A]] 0
; CHECK: %[[#P0:]] = OpConvertUToPtr %[[#PTR]] %[[#E0]]
; CHECK: %[[#E1:]] = OpCompositeExtract %[[#I64]] %[[#A]] 1
; CHECK: %[[#P1:]] = OpConvertUToPtr %[[#PTR]] %[[#E1]]
; CHECK: OpLoad %[[#I8]] %[[#P0]]
; CHECK: OpLoad %[[#I8]] %[[#P1]]
; CHECK: OpStore %[[#P0]]

define spir_kernel void @test_inttoptr_scalarized(<2 x i64> %addr) {
entry:
  %ptrs = inttoptr <2 x i64> %addr to <2 x ptr addrspace(1)>
  %p0 = extractelement <2 x ptr addrspace(1)> %ptrs, i32 0
  %p1 = extractelement <2 x ptr addrspace(1)> %ptrs, i32 1
  %val0 = load i8, ptr addrspace(1) %p0
  %val1 = load i8, ptr addrspace(1) %p1
  %sum = add i8 %val0, %val1
  store i8 %sum, ptr addrspace(1) %p0
  ret void
}

; Test scalarization with a 4-element vector.
; CHECK-LABEL: Begin function test_inttoptr_v4
; CHECK-NEXT: %[[#A:]] = OpFunctionParameter
; CHECK: %[[#E0:]] = OpCompositeExtract %[[#I64]] %[[#A]] 0
; CHECK: %[[#P0:]] = OpConvertUToPtr %[[#PTR]] %[[#E0]]
; CHECK: %[[#E1:]] = OpCompositeExtract %[[#I64]] %[[#A]] 1
; CHECK: %[[#P1:]] = OpConvertUToPtr %[[#PTR]] %[[#E1]]
; CHECK: %[[#E2:]] = OpCompositeExtract %[[#I64]] %[[#A]] 2
; CHECK: %[[#P2:]] = OpConvertUToPtr %[[#PTR]] %[[#E2]]
; CHECK: %[[#E3:]] = OpCompositeExtract %[[#I64]] %[[#A]] 3
; CHECK: %[[#P3:]] = OpConvertUToPtr %[[#PTR]] %[[#E3]]
; CHECK: OpStore %[[#P0]]
; CHECK: OpStore %[[#P1]]
; CHECK: OpStore %[[#P2]]
; CHECK: OpStore %[[#P3]]

define spir_kernel void @test_inttoptr_v4(<4 x i64> %addr) {
entry:
  %ptrs = inttoptr <4 x i64> %addr to <4 x ptr addrspace(1)>
  %p0 = extractelement <4 x ptr addrspace(1)> %ptrs, i32 0
  %p1 = extractelement <4 x ptr addrspace(1)> %ptrs, i32 1
  %p2 = extractelement <4 x ptr addrspace(1)> %ptrs, i32 2
  %p3 = extractelement <4 x ptr addrspace(1)> %ptrs, i32 3
  store i8 1, ptr addrspace(1) %p0
  store i8 2, ptr addrspace(1) %p1
  store i8 3, ptr addrspace(1) %p2
  store i8 4, ptr addrspace(1) %p3
  ret void
}

; CHECK-LABEL: Begin function test_inttoptr_partial_use
; CHECK-NEXT: %[[#A:]] = OpFunctionParameter
; CHECK: %[[#E0:]] = OpCompositeExtract %[[#I64]] %[[#A]] 0
; CHECK: %[[#P0:]] = OpConvertUToPtr %[[#PTR]] %[[#E0]]
; CHECK-NOT: OpConvertUToPtr
; CHECK: OpStore %[[#P0]]

define spir_kernel void @test_inttoptr_partial_use(<2 x i64> %addr) {
entry:
  %ptrs = inttoptr <2 x i64> %addr to <2 x ptr addrspace(1)>
  %p0 = extractelement <2 x ptr addrspace(1)> %ptrs, i32 0
  store i8 42, ptr addrspace(1) %p0
  ret void
}

; Test that an inttoptr without uses is removed.
; CHECK-LABEL: Begin function test_inttoptr_no_use
; CHECK-NOT: OpConvertUToPtr
; CHECK: OpFunctionEnd

define spir_kernel void @test_inttoptr_no_use(<2 x i64> %addr) {
entry:
  %ptrs = inttoptr <2 x i64> %addr to <2 x ptr addrspace(1)>
  ret void
}
