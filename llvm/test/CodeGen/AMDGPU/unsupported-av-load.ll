; RUN: not llc -global-isel=0 -mtriple=amdgpu6.02 -filetype=null < %s 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: not llc -global-isel=0 -mtriple=amdgpu7.05 -filetype=null < %s 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: not llc -global-isel=0 -mtriple=amdgpu8.10 -filetype=null < %s 2>&1 | FileCheck -check-prefix=ERR %s

; RUN: not llc -global-isel=1 -mtriple=amdgpu6.02 -filetype=null < %s 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: not llc -global-isel=1 -mtriple=amdgpu7.05 -filetype=null < %s 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: not llc -global-isel=1 -mtriple=amdgpu8.10 -filetype=null < %s 2>&1 | FileCheck -check-prefix=ERR %s

define <4 x i32> @av_load_b128(ptr addrspace(1) %addr) {
; ERR: error: {{.*}}: in function av_load_b128 {{.*}}: llvm.amdgcn.av.load.b128 not supported on subtarget
entry:
  %data = call <4 x i32> @llvm.amdgcn.av.load.b128.p1(ptr addrspace(1) %addr, metadata !0)
  ret <4 x i32> %data
}

!0 = !{!""}
