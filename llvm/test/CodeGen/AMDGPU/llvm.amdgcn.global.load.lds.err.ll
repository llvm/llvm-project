; RUN: not --crash llc -filetype=null -mtriple=amdgpu8.10 %s 2>&1 | FileCheck --ignore-case %s
; RUN: not llc -filetype=null -global-isel -mtriple=amdgpu8.10 %s 2>&1 | FileCheck --ignore-case %s
; RUN: not --crash llc -filetype=null -mtriple=amdgpu11.00 %s 2>&1 | FileCheck --ignore-case %s
; RUN: not llc -filetype=null -global-isel -mtriple=amdgpu11.00 %s 2>&1 | FileCheck --ignore-case %s
;
; CHECK: LLVM ERROR: Cannot select

declare void @llvm.amdgcn.global.load.lds(ptr addrspace(1) nocapture %gptr, ptr addrspace(3) nocapture %lptr, i32 %size, i32 %offset, i32 %aux)

define amdgpu_ps void @global_load_lds_dword(ptr addrspace(1) nocapture %gptr, ptr addrspace(3) nocapture %lptr) {
  call void @llvm.amdgcn.global.load.lds(ptr addrspace(1) %gptr, ptr addrspace(3) %lptr, i32 4, i32 0, i32 0)
  ret void
}
