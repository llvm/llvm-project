; RUN: sed 's/LDS_LIMIT/131072/' %s | not llc -mtriple=amdgpu13.10-amd-amdhsa 2>&1 | FileCheck -check-prefix=ERROR-128K %s
; RUN: sed 's/LDS_LIMIT/65536/' %s | not llc -mtriple=amdgpu13.10-amd-amdhsa 2>&1 | FileCheck -check-prefix=ERROR-64K %s
; RUN: sed 's/LDS_LIMIT/131072/' %s | not llc -mtriple=amdgpu13.10-amd-amdhsa -mattr=+cumode 2>&1 | FileCheck -check-prefix=ERROR-128K-CU %s
; RUN: sed 's/LDS_LIMIT/65536/' %s | not llc -mtriple=amdgpu13.10-amd-amdhsa -mattr=+cumode 2>&1 | FileCheck -check-prefix=ERROR-64K-CU %s

; ERROR-128K: error: <unknown>:0:0: local memory (131076) exceeds limit (131072) in function 'test_lds_limit'
; ERROR-64K: error: <unknown>:0:0: local memory (131076) exceeds limit (65536) in function 'test_lds_limit'
; ERROR-128K-CU: error: <unknown>:0:0: local memory (131076) exceeds limit (65536) in function 'test_lds_limit'
; ERROR-64K-CU: error: <unknown>:0:0: local memory (131076) exceeds limit (32768) in function 'test_lds_limit'
@dst = addrspace(3) global [131076 x i8] poison

define amdgpu_kernel void @test_lds_limit(i8 %val) {
  %gep = getelementptr [131076 x i8], ptr addrspace(3) @dst, i32 0, i32 100
  store i8 %val, ptr addrspace(3) %gep
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdgpu.lds.size.limit", i32 LDS_LIMIT}
