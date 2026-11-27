; Test that llvm.masked.scatter is scalarized into individual stores when the
; SPV_INTEL_masked_gather_scatter extension is not enabled.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

declare void @llvm.masked.scatter.v4i32.v4p1(<4 x i32>, <4 x ptr addrspace(1)>, i32, <4 x i1>)

; CHECK-LABEL: Begin function test_scatter_no_ext
; CHECK-NEXT: %[[#DATA:]] = OpFunctionParameter
; CHECK-NEXT: %[[#ADDRS:]] = OpFunctionParameter
; CHECK: %[[#D0:]] = OpCompositeExtract %[[#]] %[[#DATA]] 0
; CHECK: %[[#A0:]] = OpCompositeExtract %[[#]] %[[#ADDRS]] 0
; CHECK: %[[#P0:]] = OpConvertUToPtr %[[#]] %[[#A0]]
; CHECK: OpStore %[[#P0]] %[[#D0]]
; CHECK-NOT: OpStore
; CHECK: %[[#D2:]] = OpCompositeExtract %[[#]] %[[#DATA]] 2
; CHECK: %[[#A2:]] = OpCompositeExtract %[[#]] %[[#ADDRS]] 2
; CHECK: %[[#P2:]] = OpConvertUToPtr %[[#]] %[[#A2]]
; CHECK: OpStore %[[#P2]] %[[#D2]]
; CHECK-NOT: OpStore
; CHECK: OpReturn

define spir_kernel void @test_scatter_no_ext(<4 x i32> %data, <4 x i64> %addrs) {
entry:
  %ptrs = inttoptr <4 x i64> %addrs to <4 x ptr addrspace(1)>
  call void @llvm.masked.scatter.v4i32.v4p1(<4 x i32> %data, <4 x ptr addrspace(1)> %ptrs, i32 4, <4 x i1> <i1 true, i1 false, i1 true, i1 false>)
  ret void
}
