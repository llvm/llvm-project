; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -verify-machineinstrs < %s 2>&1 | FileCheck -check-prefix=ERROR %s

; GFX950 supports upto 160 KB LDS memory.
; This is a negative test to check when the LDS size exceeds the max usable limit.

; ERROR: error: <unknown>:0:0: local memory (163844) exceeds limit (163840) in function 'test_lds_limit'
@dst = addrspace(3) global [40961 x i32] poison

define amdgpu_kernel void @test_lds_limit(i32 %val) {
  %gep = getelementptr [40961 x i32], ptr addrspace(3) @dst, i32 0, i32 100
  store i32 %val, ptr addrspace(3) %gep
  ret void
}