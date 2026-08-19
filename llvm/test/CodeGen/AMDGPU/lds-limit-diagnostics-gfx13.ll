; RUN: not llc -mtriple=amdgpu13.10-amd-amdhsa -verify-machineinstrs < %s 2>&1 | FileCheck -check-prefix=ERROR %s
; RUN: not llc -mtriple=amdgpu13.10-amd-amdhsa -mattr=+cumode -verify-machineinstrs < %s 2>&1 | FileCheck -check-prefix=ERROR-CU %s

; GFX1310 has up to 192 KB LDS when a work-group runs on all four SIMD32s, where
; the whole block is addressable by a single work-group. When it runs on only two
; SIMD32s the block is split between them, so only half (96 KB) is usable.
; These are negative tests checking when the LDS size exceeds the usable limit.

; ERROR: error: <unknown>:0:0: local memory (196612) exceeds limit (196608) in function 'test_lds_limit'
; ERROR-CU: error: <unknown>:0:0: local memory (196612) exceeds limit (98304) in function 'test_lds_limit'
@dst = addrspace(3) global [196612 x i8] poison

define amdgpu_kernel void @test_lds_limit(i8 %val) {
  %gep = getelementptr [196612 x i8], ptr addrspace(3) @dst, i32 0, i32 100
  store i8 %val, ptr addrspace(3) %gep
  ret void
}
