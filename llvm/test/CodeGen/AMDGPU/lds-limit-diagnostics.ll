; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -filetype=null %s 2>&1 | FileCheck -check-prefix=ERROR-LIMIT160K %s
; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx9-4-generic -filetype=null %s 2>&1 | FileCheck -check-prefix=ERROR-LIMIT64K %s
; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx9-generic -filetype=null %s 2>&1 | FileCheck -check-prefix=ERROR-LIMIT64K %s
; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx940 -filetype=null %s 2>&1 | FileCheck -check-prefix=ERROR-LIMIT64K %s
; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx941 -filetype=null %s 2>&1 | FileCheck -check-prefix=ERROR-LIMIT64K %s
; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -filetype=null %s 2>&1 | FileCheck -check-prefix=ERROR-LIMIT64K %s
; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=null %s 2>&1 | FileCheck -check-prefix=ERROR-LIMIT64K %s
; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx906 -filetype=null %s 2>&1 | FileCheck -check-prefix=ERROR-LIMIT64K %s
; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=null %s 2>&1 | FileCheck -check-prefix=ERROR-LIMIT64K %s
; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a -filetype=null %s 2>&1 | FileCheck -check-prefix=ERROR-LIMIT64K %s
; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90c -filetype=null %s 2>&1 | FileCheck -check-prefix=ERROR-LIMIT64K %s
; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -filetype=null %s 2>&1 | FileCheck -check-prefix=ERROR-LIMIT64K %s
; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1030 -filetype=null %s 2>&1 | FileCheck -check-prefix=ERROR-LIMIT64K %s
; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 -filetype=null %s 2>&1 | FileCheck -check-prefix=ERROR-LIMIT64K %s
; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1200 -filetype=null %s 2>&1 | FileCheck -check-prefix=ERROR-LIMIT64K %s
; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx803 -filetype=null %s 2>&1 | FileCheck -check-prefix=ERROR-LIMIT64K %s
; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 -filetype=null %s 2>&1 | FileCheck -check-prefix=ERROR-LIMIT64K %s
; RUN: not llc -mtriple=amdgcn-amd-amdpal -mcpu=gfx600 -filetype=null %s 2>&1 | FileCheck -check-prefix=ERROR-LIMIT32K %s

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
