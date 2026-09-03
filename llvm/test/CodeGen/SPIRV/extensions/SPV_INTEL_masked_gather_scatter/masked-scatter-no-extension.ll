; Test that llvm.masked.scatter is scalarized into individual stores when the
; SPV_INTEL_masked_gather_scatter extension is not enabled. The generic
; ScalarizeMaskedMemIntrin pass handles this before SPIR-V-specific passes run.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

declare void @llvm.masked.scatter.v4i32.v4p1(<4 x i32>, <4 x ptr addrspace(1)>, i32, <4 x i1>)

; CHECK: OpFunction
; CHECK: OpStore
; CHECK: OpStore
; CHECK: OpReturn

define spir_kernel void @test_scatter_no_ext(<4 x i32> %data, <4 x i64> %addrs) {
entry:
  %ptrs = inttoptr <4 x i64> %addrs to <4 x ptr addrspace(1)>
  call void @llvm.masked.scatter.v4i32.v4p1(<4 x i32> %data, <4 x ptr addrspace(1)> %ptrs, i32 4, <4 x i1> <i1 true, i1 false, i1 true, i1 false>)
  ret void
}
