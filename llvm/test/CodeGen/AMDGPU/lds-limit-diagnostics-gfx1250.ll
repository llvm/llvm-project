; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 < %s 2>&1 | FileCheck -check-prefix=ERROR %s

; GFX1250 supports upto 320 KB LDS memory.
; This is a negative test to check when the LDS size exceeds the max usable limit.

; ERROR: error: <unknown>:0:0: local memory (327684) exceeds limit (327680) in function 'test_lds_limit'
@dst = addrspace(3) global [81921 x i32] undef

define amdgpu_kernel void @test_lds_limit(i32 %val) {
  %gep = getelementptr [81921 x i32], ptr addrspace(3) @dst, i32 0, i32 100
  store i32 %val, ptr addrspace(3) %gep
  ret void
}
