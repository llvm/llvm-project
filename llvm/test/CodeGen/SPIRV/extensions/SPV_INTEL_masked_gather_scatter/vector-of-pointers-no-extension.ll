; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: error:{{.*}}Vector of pointers requires SPV_INTEL_masked_gather_scatter extension

declare spir_func void @foo(<2 x i64>)

define spir_kernel void @test_ptrtoint(<2 x ptr addrspace(1)> %p) {
entry:
  %addr = ptrtoint <2 x ptr addrspace(1)> %p to <2 x i64>
  call void @foo(<2 x i64> %addr)
  ret void
}
