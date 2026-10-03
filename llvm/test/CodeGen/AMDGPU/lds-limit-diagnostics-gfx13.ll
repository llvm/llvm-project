; RUN: not llc -mtriple=amdgpu13.10-amd-amdhsa < %s 2>&1 | FileCheck -check-prefix=ERROR %s
; RUN: not llc -mtriple=amdgpu13.10-amd-amdhsa -mattr=+cumode < %s 2>&1 | FileCheck -check-prefix=ERROR-CU %s

; GFX1310 has up to 192 KB LDS when a work-group runs on all four SIMD32s. Then
; one work-group can address the whole block. On only two SIMD32s the block is
; split between them, so only half (96 KB) can be used.
; Negative tests, they check when the LDS size is over the usable limit.

; ERROR: error: <unknown>:0:0: local memory (196612) exceeds limit (196608) in function 'test_lds_limit'
; ERROR-CU: error: <unknown>:0:0: local memory (196612) exceeds limit (98304) in function 'test_lds_limit'
@dst = addrspace(3) global [196612 x i8] poison

define amdgpu_kernel void @test_lds_limit(i8 %val) {
  %gep = getelementptr [196612 x i8], ptr addrspace(3) @dst, i32 0, i32 100
  store i8 %val, ptr addrspace(3) %gep
  ret void
}
