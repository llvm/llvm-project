; RUN: not llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx1010 -filetype=null %s 2>&1 | FileCheck %s
; RUN: not llc -global-isel=1 -mtriple=amdgcn -mcpu=gfx1010 -filetype=null %s 2>&1 | FileCheck %s

; CHECK: error: <unknown>:0:0: in function ds_ordered_add_dword_count_too_low void (ptr addrspace(2), ptr addrspace(1)): ds_ordered_count: dword count must be between 1 and 4
define amdgpu_kernel void @ds_ordered_add_dword_count_too_low(ptr addrspace(2) inreg %gds, ptr addrspace(1) %out) {
  %val = call i32 @llvm.amdgcn.ds.ordered.add(ptr addrspace(2) %gds, i32 31, i32 0, i32 0, i1 false, i32 0, i1 true, i1 true)
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; CHECK: error: <unknown>:0:0: in function ds_ordered_add_dword_count_too_high void (ptr addrspace(2), ptr addrspace(1)): ds_ordered_count: dword count must be between 1 and 4
define amdgpu_kernel void @ds_ordered_add_dword_count_too_high(ptr addrspace(2) inreg %gds, ptr addrspace(1) %out) {
  %val = call i32 @llvm.amdgcn.ds.ordered.add(ptr addrspace(2) %gds, i32 31, i32 0, i32 0, i1 false, i32 5, i1 true, i1 true)
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; CHECK: error: <unknown>:0:0: in function ds_ordered_add_bad_index_operand void (ptr addrspace(2), ptr addrspace(1)): ds_ordered_count: bad index operand
define amdgpu_kernel void @ds_ordered_add_bad_index_operand(ptr addrspace(2) inreg %gds, ptr addrspace(1) %out) {
  %val = call i32 @llvm.amdgcn.ds.ordered.add(ptr addrspace(2) %gds, i32 31, i32 0, i32 1, i1 false, i32 -1, i1 true, i1 true)
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; CHECK: error: <unknown>:0:0: in function ds_ordered_add_dword_count_wave_done_without_wave_release void (ptr addrspace(2), ptr addrspace(1)): ds_ordered_count: wave_done requires wave_release
define amdgpu_kernel void @ds_ordered_add_dword_count_wave_done_without_wave_release(ptr addrspace(2) inreg %gds, ptr addrspace(1) %out) {
  %val = call i32 @llvm.amdgcn.ds.ordered.add(ptr addrspace(2) %gds, i32 31, i32 0, i32 0, i1 false, i32 1, i1 false, i1 true)
  store i32 %val, ptr addrspace(1) %out
  ret void
}
