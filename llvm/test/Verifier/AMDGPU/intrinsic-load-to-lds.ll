; RUN: not llvm-as %s -disable-output 2>&1 | FileCheck %s

declare void @llvm.amdgcn.global.load.lds(ptr addrspace(1), ptr addrspace(3), i32, i32, i32)
declare void @llvm.amdgcn.load.to.lds(ptr addrspace(1), ptr addrspace(3), i32, i32, i32)
declare void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8), ptr addrspace(3), i32, i32, i32, i32, i32)

define void @global_load_lds_size_zero(ptr addrspace(1) %gptr, ptr addrspace(3) %lptr) {
  ; CHECK: invalid data size for load-to-LDS intrinsic; must be 1, 2, 4, 12, or 16
  call void @llvm.amdgcn.global.load.lds(ptr addrspace(1) %gptr, ptr addrspace(3) %lptr, i32 0, i32 0, i32 0)
  ret void
}

define void @load_to_lds_size_three(ptr addrspace(1) %gptr, ptr addrspace(3) %lptr) {
  ; CHECK: invalid data size for load-to-LDS intrinsic; must be 1, 2, 4, 12, or 16
  call void @llvm.amdgcn.load.to.lds(ptr addrspace(1) %gptr, ptr addrspace(3) %lptr, i32 3, i32 0, i32 0)
  ret void
}

define void @raw_ptr_buffer_load_lds_size_eight(ptr addrspace(8) inreg %rsrc, ptr addrspace(3) inreg %lds) {
  ; CHECK: invalid data size for load-to-LDS intrinsic; must be 1, 2, 4, 12, or 16
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %rsrc, ptr addrspace(3) %lds, i32 8, i32 0, i32 0, i32 0, i32 0)
  ret void
}
