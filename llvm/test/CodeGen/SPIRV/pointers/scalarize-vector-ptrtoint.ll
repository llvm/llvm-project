; Test that ptrtoint on a vector of pointers to a vector of integers is
; scalarized into per-element ptrtoint operations, since SPIR-V does not
; support vectors of pointers without extensions.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

@g = addrspace(1) global i32 0

declare spir_func void @foo(<2 x i64>)
declare spir_func void @bar(<4 x i64>)

; CHECK-DAG: %[[#I64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#V2I64:]] = OpTypeVector %[[#I64]] 2
; CHECK-DAG: %[[#V4I64:]] = OpTypeVector %[[#I64]] 4
; CHECK-DAG: %[[#GV:]] = OpVariable %[[#]] CrossWorkgroup
; CHECK-DAG: %[[#G:]] = OpSpecConstantOp %[[#I64]] ConvertPtrToU %[[#GV]]

; CHECK-LABEL: Begin function test_ptrtoint_scalarized
; CHECK-NEXT: %[[#P0:]] = OpFunctionParameter
; CHECK-NEXT: %[[#P1:]] = OpFunctionParameter
; CHECK: %[[#C0:]] = OpConvertPtrToU %[[#I64]] %[[#P0]]
; CHECK: %[[#V0:]] = OpCompositeInsert %[[#V2I64]] %[[#C0]] %[[#]] 0
; CHECK: %[[#C1:]] = OpConvertPtrToU %[[#I64]] %[[#P1]]
; CHECK: %[[#V1:]] = OpCompositeInsert %[[#V2I64]] %[[#C1]] %[[#V0]] 1
; CHECK: OpFunctionCall %[[#]] %[[#]] %[[#V1]]

define spir_kernel void @test_ptrtoint_scalarized(ptr addrspace(1) %p0, ptr addrspace(1) %p1) {
entry:
  %vec0 = insertelement <2 x ptr addrspace(1)> poison, ptr addrspace(1) %p0, i32 0
  %vec = insertelement <2 x ptr addrspace(1)> %vec0, ptr addrspace(1) %p1, i32 1
  %addr = ptrtoint <2 x ptr addrspace(1)> %vec to <2 x i64>
  call void @foo(<2 x i64> %addr)
  ret void
}

; Test ptrtoint scalarization with a 4-element vector.
; CHECK-LABEL: Begin function test_ptrtoint_v4
; CHECK-NEXT: %[[#P0:]] = OpFunctionParameter
; CHECK-NEXT: %[[#P1:]] = OpFunctionParameter
; CHECK-NEXT: %[[#P2:]] = OpFunctionParameter
; CHECK-NEXT: %[[#P3:]] = OpFunctionParameter
; CHECK: %[[#C0:]] = OpConvertPtrToU %[[#I64]] %[[#P0]]
; CHECK: %[[#V0:]] = OpCompositeInsert %[[#V4I64]] %[[#C0]] %[[#]] 0
; CHECK: %[[#C1:]] = OpConvertPtrToU %[[#I64]] %[[#P1]]
; CHECK: %[[#V1:]] = OpCompositeInsert %[[#V4I64]] %[[#C1]] %[[#V0]] 1
; CHECK: %[[#C2:]] = OpConvertPtrToU %[[#I64]] %[[#P2]]
; CHECK: %[[#V2:]] = OpCompositeInsert %[[#V4I64]] %[[#C2]] %[[#V1]] 2
; CHECK: %[[#C3:]] = OpConvertPtrToU %[[#I64]] %[[#P3]]
; CHECK: %[[#V3:]] = OpCompositeInsert %[[#V4I64]] %[[#C3]] %[[#V2]] 3
; CHECK: OpFunctionCall %[[#]] %[[#]] %[[#V3]]

define spir_kernel void @test_ptrtoint_v4(ptr addrspace(1) %p0, ptr addrspace(1) %p1, ptr addrspace(1) %p2, ptr addrspace(1) %p3) {
entry:
  %vec0 = insertelement <4 x ptr addrspace(1)> poison, ptr addrspace(1) %p0, i32 0
  %vec1 = insertelement <4 x ptr addrspace(1)> %vec0, ptr addrspace(1) %p1, i32 1
  %vec2 = insertelement <4 x ptr addrspace(1)> %vec1, ptr addrspace(1) %p2, i32 2
  %vec3 = insertelement <4 x ptr addrspace(1)> %vec2, ptr addrspace(1) %p3, i32 3
  %addr = ptrtoint <4 x ptr addrspace(1)> %vec3 to <4 x i64>
  call void @bar(<4 x i64> %addr)
  ret void
}

; Test that a lane not set by insertelement is taken from the constant base.
; CHECK-LABEL: Begin function test_ptrtoint_const_lane
; CHECK-NEXT: %[[#P0:]] = OpFunctionParameter
; CHECK: %[[#C0:]] = OpConvertPtrToU %[[#I64]] %[[#P0]]
; CHECK: %[[#V0:]] = OpCompositeInsert %[[#V2I64]] %[[#C0]] %[[#]] 0
; CHECK: %[[#V1:]] = OpCompositeInsert %[[#V2I64]] %[[#G]] %[[#V0]] 1
; CHECK: OpFunctionCall %[[#]] %[[#]] %[[#V1]]

define spir_kernel void @test_ptrtoint_const_lane(ptr addrspace(1) %p0) {
entry:
  %vec = insertelement <2 x ptr addrspace(1)> <ptr addrspace(1) null, ptr addrspace(1) @g>, ptr addrspace(1) %p0, i32 0
  %addr = ptrtoint <2 x ptr addrspace(1)> %vec to <2 x i64>
  call void @foo(<2 x i64> %addr)
  ret void
}
