; Test that llvm.masked.gather is scalarized into individual loads when the
; SPV_INTEL_masked_gather_scatter extension is not enabled. The generic
; ScalarizeMaskedMemIntrin pass handles this before SPIR-V-specific passes run.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

declare <4 x i32> @llvm.masked.gather.v4i32.v4p1(<4 x ptr addrspace(1)>, i32, <4 x i1>, <4 x i32>)

; CHECK: OpFunction
; CHECK: OpLoad
; CHECK: OpLoad
; CHECK: OpLoad
; CHECK: OpLoad

define spir_kernel void @test_gather_no_ext(<4 x i64> %addrs, ptr addrspace(1) %out) {
entry:
  %ptrs = inttoptr <4 x i64> %addrs to <4 x ptr addrspace(1)>
  %data = call <4 x i32> @llvm.masked.gather.v4i32.v4p1(<4 x ptr addrspace(1)> %ptrs, i32 4, <4 x i1> <i1 true, i1 true, i1 true, i1 true>, <4 x i32> zeroinitializer)
  store <4 x i32> %data, ptr addrspace(1) %out
  ret void
}
