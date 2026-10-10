; RUN: not llc -mtriple=amdgpu13.10-amd-amdhsa -mattr=+local-memory-size-limit-131072 < %s 2>&1 | FileCheck -check-prefix=ERROR-128K %s
; RUN: not llc -mtriple=amdgpu13.10-amd-amdhsa -mattr=+local-memory-size-limit-65536 < %s 2>&1 | FileCheck -check-prefix=ERROR-64K %s
; RUN: not llc -mtriple=amdgpu13.10-amd-amdhsa -mattr=+local-memory-size-limit-131072,+cumode < %s 2>&1 | FileCheck -check-prefix=ERROR-128K-CU %s

; ERROR-128K: error: <unknown>:0:0: local memory (131076) exceeds limit (131072) in function 'test_lds_limit'
; ERROR-64K: error: <unknown>:0:0: local memory (131076) exceeds limit (65536) in function 'test_lds_limit'
; ERROR-128K-CU: error: <unknown>:0:0: local memory (131076) exceeds limit (65536) in function 'test_lds_limit'
@dst = addrspace(3) global [131076 x i8] poison

define amdgpu_kernel void @test_lds_limit(i8 %val) {
  %gep = getelementptr [131076 x i8], ptr addrspace(3) @dst, i32 0, i32 100
  store i8 %val, ptr addrspace(3) %gep
  ret void
}
