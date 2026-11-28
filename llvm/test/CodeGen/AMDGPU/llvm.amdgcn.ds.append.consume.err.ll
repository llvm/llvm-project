; RUN: not llc -global-isel=0 -mtriple=amdgpu12.00 -filetype=null %s 2>&1 | FileCheck %s
; RUN: not llc -global-isel=1 -mtriple=amdgpu12.00 -filetype=null %s 2>&1 | FileCheck %s

; CHECK: error: <unknown>:0:0: in function ds_append_gds_unsupported void (ptr addrspace(2), ptr addrspace(1)): GDS is not supported on this subtarget
; CHECK-NOT: error:
define amdgpu_kernel void @ds_append_gds_unsupported(ptr addrspace(2) %gds, ptr addrspace(1) %out) {
  %val = call i32 @llvm.amdgcn.ds.append.p2(ptr addrspace(2) %gds, i1 false)
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; CHECK: error: <unknown>:0:0: in function ds_append_lds_non_constant void (ptr addrspace(3), ptr addrspace(1)): ds_append/ds_consume requires a constant, dword aligned LDS address that fits in the 16 bit offset field
define amdgpu_kernel void @ds_append_lds_non_constant(ptr addrspace(3) %lds, ptr addrspace(1) %out) {
  %val = call i32 @llvm.amdgcn.ds.append.p3(ptr addrspace(3) %lds, i1 false)
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; CHECK: error: <unknown>:0:0: in function ds_append_lds_misaligned void (ptr addrspace(1)): ds_append/ds_consume requires a constant, dword aligned LDS address that fits in the 16 bit offset field
define amdgpu_kernel void @ds_append_lds_misaligned(ptr addrspace(1) %out) {
  %val = call i32 @llvm.amdgcn.ds.append.p3(ptr addrspace(3) inttoptr (i32 2 to ptr addrspace(3)), i1 false)
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; CHECK: error: <unknown>:0:0: in function ds_append_lds_over_max_offset void (ptr addrspace(1)): ds_append/ds_consume requires a constant, dword aligned LDS address that fits in the 16 bit offset field
; CHECK-NOT: error:
define amdgpu_kernel void @ds_append_lds_over_max_offset(ptr addrspace(1) %out) {
  %val = call i32 @llvm.amdgcn.ds.append.p3(ptr addrspace(3) inttoptr (i32 65536 to ptr addrspace(3)), i1 false)
  store i32 %val, ptr addrspace(1) %out
  ret void
}
