; RUN: not llc -global-isel=0 -mtriple=amdgpu6.02 -filetype=null < %s 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: not llc -global-isel=0 -mtriple=amdgpu7.05 -filetype=null < %s 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: not llc -global-isel=0 -mtriple=amdgpu8.10 -filetype=null < %s 2>&1 | FileCheck -check-prefix=ERR %s

; RUN: not llc -global-isel=1 -mtriple=amdgpu6.02 -filetype=null < %s 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: not llc -global-isel=1 -mtriple=amdgpu7.05 -filetype=null < %s 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: not llc -global-isel=1 -mtriple=amdgpu8.10 -filetype=null < %s 2>&1 | FileCheck -check-prefix=ERR %s

define void @av_store_b128(ptr addrspace(1) %addr, <4 x i32> %data) {
; ERR: error: {{.*}}: in function av_store_b128 {{.*}}: llvm.amdgcn.av.store.b128 not supported on subtarget
entry:
  call void @llvm.amdgcn.av.store.b128.p1(ptr addrspace(1) %addr, <4 x i32> %data, metadata !0)
  ret void
}

!0 = !{!""}
