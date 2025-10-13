; RUN: not llc -mtriple=amdgcn-amd-amdpal -mcpu=gfx900 -filetype=null 2>&1 %s | FileCheck %s

; CHECK: error: <unknown>:0:0: in function ds_ordered_add_amdgpu_hs void (ptr addrspace(2), ptr addrspace(1)): ds_ordered_count unsupported for this calling conv
define amdgpu_hs void @ds_ordered_add_amdgpu_hs(ptr addrspace(2) inreg %gds, ptr addrspace(1) %out) {
  %val = call i32 @llvm.amdgcn.ds.ordered.add(ptr addrspace(2) %gds, i32 31, i32 0, i32 0, i1 false, i32 1, i1 true, i1 true)
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; CHECK: error: <unknown>:0:0: in function ds_ordered_add_amdgpu_ls void (ptr addrspace(2), ptr addrspace(1)): ds_ordered_count unsupported for this calling conv
define amdgpu_ls void @ds_ordered_add_amdgpu_ls(ptr addrspace(2) inreg %gds, ptr addrspace(1) %out) {
  %val = call i32 @llvm.amdgcn.ds.ordered.add(ptr addrspace(2) %gds, i32 31, i32 0, i32 0, i1 false, i32 1, i1 true, i1 true)
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; CHECK: error: <unknown>:0:0: in function ds_ordered_add_amdgpu_es void (ptr addrspace(2), ptr addrspace(1)): ds_ordered_count unsupported for this calling conv
define amdgpu_es void @ds_ordered_add_amdgpu_es(ptr addrspace(2) inreg %gds, ptr addrspace(1) %out) {
  %val = call i32 @llvm.amdgcn.ds.ordered.add(ptr addrspace(2) %gds, i32 31, i32 0, i32 0, i1 false, i32 1, i1 true, i1 true)
  store i32 %val, ptr addrspace(1) %out
  ret void
}
