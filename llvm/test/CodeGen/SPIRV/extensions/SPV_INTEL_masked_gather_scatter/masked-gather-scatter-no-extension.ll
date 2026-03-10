; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown %s -o - 2>&1 | FileCheck %s

; CHECK: error:{{.*}}: llvm.masked.gather requires SPV_INTEL_masked_gather_scatter

define spir_kernel void @test_gather_no_ext(<4 x i64> %addrs, <4 x i1> %mask, <4 x i32> %passthru) {
entry:
  %ptrs = inttoptr <4 x i64> %addrs to <4 x ptr addrspace(1)>
  %data = call <4 x i32> @llvm.masked.gather.v4i32.v4p1(<4 x ptr addrspace(1)> %ptrs, i32 4, <4 x i1> %mask, <4 x i32> %passthru)
  ret void
}

declare <4 x i32> @llvm.masked.gather.v4i32.v4p1(<4 x ptr addrspace(1)>, i32, <4 x i1>, <4 x i32>)
