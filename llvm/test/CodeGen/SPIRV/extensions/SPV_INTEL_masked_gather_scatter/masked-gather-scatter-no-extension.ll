; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: error: llvm.masked.gather requires SPV_INTEL_masked_gather_scatter extension

define spir_kernel void @test_gather_no_ext(<4 x ptr addrspace(1)> %ptrs, <4 x i1> %mask, <4 x i32> %passthru) {
entry:
  %data = call <4 x i32> @llvm.masked.gather.v4i32.v4p1(<4 x ptr addrspace(1)> %ptrs, i32 4, <4 x i1> %mask, <4 x i32> %passthru)
  ret void
}

declare <4 x i32> @llvm.masked.gather.v4i32.v4p1(<4 x ptr addrspace(1)>, i32, <4 x i1>, <4 x i32>)

; CHECK: error: llvm.masked.scatter requires SPV_INTEL_masked_gather_scatter extension

define spir_kernel void @test_scatter_no_ext(<4 x i32> %data, <4 x ptr addrspace(1)> %ptrs, <4 x i1> %mask) {
entry:
  call void @llvm.masked.scatter.v4i32.v4p1(<4 x i32> %data, <4 x ptr addrspace(1)> %ptrs, i32 4, <4 x i1> %mask)
  ret void
}

declare void @llvm.masked.scatter.v4i32.v4p1(<4 x i32>, <4 x ptr addrspace(1)>, i32, <4 x i1>)
