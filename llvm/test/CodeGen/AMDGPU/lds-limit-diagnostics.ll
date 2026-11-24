; RUN: not llc -mtriple=amdgpu9.50-amd-amdhsa -filetype=null %s 2>&1 | FileCheck -check-prefix=ERROR-LIMIT160K %s
; RUN: not llc -mtriple=amdgpu9.4-amd-amdhsa -filetype=null %s 2>&1 | FileCheck -check-prefix=ERROR-LIMIT64K %s
; RUN: not llc -mtriple=amdgpu9-amd-amdhsa -filetype=null %s 2>&1 | FileCheck -check-prefix=ERROR-LIMIT64K %s
; RUN: not llc -mtriple=amdgpu9.42-amd-amdhsa -filetype=null %s 2>&1 | FileCheck -check-prefix=ERROR-LIMIT64K %s
; RUN: not llc -mtriple=amdgpu9.00-amd-amdhsa -filetype=null %s 2>&1 | FileCheck -check-prefix=ERROR-LIMIT64K %s
; RUN: not llc -mtriple=amdgpu9.06-amd-amdhsa -filetype=null %s 2>&1 | FileCheck -check-prefix=ERROR-LIMIT64K %s
; RUN: not llc -mtriple=amdgpu9.08-amd-amdhsa -filetype=null %s 2>&1 | FileCheck -check-prefix=ERROR-LIMIT64K %s
; RUN: not llc -mtriple=amdgpu9.0a-amd-amdhsa -filetype=null %s 2>&1 | FileCheck -check-prefix=ERROR-LIMIT64K %s
; RUN: not llc -mtriple=amdgpu9.0c-amd-amdhsa -filetype=null %s 2>&1 | FileCheck -check-prefix=ERROR-LIMIT64K %s
; RUN: not llc -mtriple=amdgpu10.10-amd-amdhsa -filetype=null %s 2>&1 | FileCheck -check-prefix=ERROR-LIMIT64K %s
; RUN: not llc -mtriple=amdgpu10.30-amd-amdhsa -filetype=null %s 2>&1 | FileCheck -check-prefix=ERROR-LIMIT64K %s
; RUN: not llc -mtriple=amdgpu11.00-amd-amdhsa -filetype=null %s 2>&1 | FileCheck -check-prefix=ERROR-LIMIT64K %s
; RUN: not llc -mtriple=amdgpu12.00-amd-amdhsa -filetype=null %s 2>&1 | FileCheck -check-prefix=ERROR-LIMIT64K %s
; RUN: not llc -mtriple=amdgpu8.03-amd-amdhsa -filetype=null %s 2>&1 | FileCheck -check-prefix=ERROR-LIMIT64K %s
; RUN: not llc -mtriple=amdgpu7.00-amd-amdhsa -filetype=null %s 2>&1 | FileCheck -check-prefix=ERROR-LIMIT64K %s
; RUN: not llc -mtriple=amdgpu6.00-amd-amdpal -filetype=null %s 2>&1 | FileCheck -check-prefix=ERROR-LIMIT32K %s

; gfx950 supports upto 160 KB LDS memory. The generic target does not.
; This is a negative test to check when the LDS size exceeds the max usable limit.

; ERROR-LIMIT160K: error: <unknown>:0:0: local memory (163844) exceeds limit (163840) in function 'test_lds_limit'
; ERROR-LIMIT64K: error: <unknown>:0:0: local memory (163844) exceeds limit (65536) in function 'test_lds_limit'
; ERROR-LIMIT32K: error: <unknown>:0:0: local memory (163844) exceeds limit (32768) in function 'test_lds_limit'
@dst = addrspace(3) global [40961 x i32] poison

define amdgpu_kernel void @test_lds_limit(i32 %val) {
  %gep = getelementptr [40961 x i32], ptr addrspace(3) @dst, i32 0, i32 100
  store i32 %val, ptr addrspace(3) %gep
  ret void
}
