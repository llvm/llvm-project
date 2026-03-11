; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s

declare <4 x i32> @llvm.masked.gather.v4i32.v4p1(<4 x ptr addrspace(1)>, i32, <4 x i1>, <4 x i32>)

; CHECK: error: {{.*}}Vector of pointers requires SPV_INTEL_masked_gather_scatter extension

define spir_kernel void @test_gather_no_ext(<4 x i64> %addrs) {
entry:
  %ptrs = inttoptr <4 x i64> %addrs to <4 x ptr addrspace(1)>
  %data = call <4 x i32> @llvm.masked.gather.v4i32.v4p1(<4 x ptr addrspace(1)> %ptrs, i32 4, <4 x i1> <i1 true, i1 true, i1 true, i1 true>, <4 x i32> zeroinitializer)
  ret void
}
