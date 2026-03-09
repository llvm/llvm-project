; REQUIRES: asserts
; RUN: not --crash llc -O0 -mtriple=spirv64-unknown-unknown %s 2>&1 | FileCheck %s

; CHECK: Vector of pointers requires SPV_INTEL_masked_gather_scatter

declare spir_func void @foo(<2 x i64>)

define spir_kernel void @test_ptrtoint(<2 x ptr addrspace(1)> %p) {
entry:
  %addr = ptrtoint <2 x ptr addrspace(1)> %p to <2 x i64>
  call void @foo(<2 x i64> %addr)
  ret void
}
