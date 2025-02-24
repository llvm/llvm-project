; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1300 -verify-machineinstrs < %s 2>&1 | FileCheck -check-prefix=ERROR %s

; GFX1300 supports upto 192 KB LDS memory.
; This is a negative test to check when the LDS size exceeds the max usable limit.

; ERROR: error: <unknown>:0:0: local memory (196612) exceeds limit (196608) in function 'test_lds_limit'
@dst = addrspace(3) global [196612 x i8] undef

define amdgpu_kernel void @test_lds_limit(i8 %val) {
  %gep = getelementptr [196612 x i8], ptr addrspace(3) @dst, i32 0, i32 100
  store i8 %val, ptr addrspace(3) %gep
  ret void
}
