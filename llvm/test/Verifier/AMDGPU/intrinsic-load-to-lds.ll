; RUN: not llvm-as %s -disable-output 2>&1 | FileCheck %s

declare void @llvm.amdgcn.load.to.lds.p1(ptr addrspace(1), ptr addrspace(3), i32, i32, i32)
declare void @llvm.amdgcn.load.async.to.lds.p1(ptr addrspace(1), ptr addrspace(3), i32, i32, i32)
declare void @llvm.amdgcn.global.load.lds(ptr addrspace(1), ptr addrspace(3), i32, i32, i32)
declare void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1), ptr addrspace(3), i32, i32, i32)
declare void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32>, ptr addrspace(3), i32, i32, i32, i32, i32)
declare void @llvm.amdgcn.raw.buffer.load.async.lds(<4 x i32>, ptr addrspace(3), i32, i32, i32, i32, i32)
declare void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8), ptr addrspace(3), i32, i32, i32, i32, i32)
declare void @llvm.amdgcn.raw.ptr.buffer.load.async.lds(ptr addrspace(8), ptr addrspace(3), i32, i32, i32, i32, i32)
declare void @llvm.amdgcn.struct.buffer.load.lds(<4 x i32>, ptr addrspace(3), i32, i32, i32, i32, i32, i32)
declare void @llvm.amdgcn.struct.buffer.load.async.lds(<4 x i32>, ptr addrspace(3), i32, i32, i32, i32, i32, i32)
declare void @llvm.amdgcn.struct.ptr.buffer.load.lds(ptr addrspace(8), ptr addrspace(3), i32, i32, i32, i32, i32, i32)
declare void @llvm.amdgcn.struct.ptr.buffer.load.async.lds(ptr addrspace(8), ptr addrspace(3), i32, i32, i32, i32, i32, i32)

define void @load_to_lds(ptr addrspace(1) %gptr, ptr addrspace(3) %lptr) {
  ; CHECK: invalid data size for load-to-LDS intrinsic; must be 1, 2, 4, 12, or 16
  call void @llvm.amdgcn.load.to.lds.p1(ptr addrspace(1) %gptr, ptr addrspace(3) %lptr, i32 0, i32 0, i32 0)
  ; CHECK: invalid data size for load-to-LDS intrinsic; must be 1, 2, 4, 12, or 16
  call void @llvm.amdgcn.load.async.to.lds.p1(ptr addrspace(1) %gptr, ptr addrspace(3) %lptr, i32 3, i32 0, i32 0)
  ; CHECK: invalid data size for load-to-LDS intrinsic; must be 1, 2, 4, 12, or 16
  call void @llvm.amdgcn.global.load.lds(ptr addrspace(1) %gptr, ptr addrspace(3) %lptr, i32 5, i32 0, i32 0)
  ; CHECK: invalid data size for load-to-LDS intrinsic; must be 1, 2, 4, 12, or 16
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %gptr, ptr addrspace(3) %lptr, i32 8, i32 0, i32 0)
  ret void
}

define void @raw_buffer_load_lds(<4 x i32> %rsrc, ptr addrspace(3) %lptr) {
  ; CHECK: invalid data size for load-to-LDS intrinsic; must be 1, 2, 4, 12, or 16
  call void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32> %rsrc, ptr addrspace(3) %lptr, i32 0, i32 0, i32 0, i32 0, i32 0)
  ; CHECK: invalid data size for load-to-LDS intrinsic; must be 1, 2, 4, 12, or 16
  call void @llvm.amdgcn.raw.buffer.load.async.lds(<4 x i32> %rsrc, ptr addrspace(3) %lptr, i32 3, i32 0, i32 0, i32 0, i32 0)
  ret void
}

define void @raw_ptr_buffer_load_lds(ptr addrspace(8) %rsrc, ptr addrspace(3) %lptr) {
  ; CHECK: invalid data size for load-to-LDS intrinsic; must be 1, 2, 4, 12, or 16
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %rsrc, ptr addrspace(3) %lptr, i32 5, i32 0, i32 0, i32 0, i32 0)
  ; CHECK: invalid data size for load-to-LDS intrinsic; must be 1, 2, 4, 12, or 16
  call void @llvm.amdgcn.raw.ptr.buffer.load.async.lds(ptr addrspace(8) %rsrc, ptr addrspace(3) %lptr, i32 8, i32 0, i32 0, i32 0, i32 0)
  ret void
}

define void @struct_buffer_load_lds(<4 x i32> %rsrc, ptr addrspace(3) %lptr) {
  ; CHECK: invalid data size for load-to-LDS intrinsic; must be 1, 2, 4, 12, or 16
  call void @llvm.amdgcn.struct.buffer.load.lds(<4 x i32> %rsrc, ptr addrspace(3) %lptr, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  ; CHECK: invalid data size for load-to-LDS intrinsic; must be 1, 2, 4, 12, or 16
  call void @llvm.amdgcn.struct.buffer.load.async.lds(<4 x i32> %rsrc, ptr addrspace(3) %lptr, i32 3, i32 0, i32 0, i32 0, i32 0, i32 0)
  ret void
}

define void @struct_ptr_buffer_load_lds(ptr addrspace(8) %rsrc, ptr addrspace(3) %lptr) {
  ; CHECK: invalid data size for load-to-LDS intrinsic; must be 1, 2, 4, 12, or 16
  call void @llvm.amdgcn.struct.ptr.buffer.load.lds(ptr addrspace(8) %rsrc, ptr addrspace(3) %lptr, i32 5, i32 0, i32 0, i32 0, i32 0, i32 0)
  ; CHECK: invalid data size for load-to-LDS intrinsic; must be 1, 2, 4, 12, or 16
  call void @llvm.amdgcn.struct.ptr.buffer.load.async.lds(ptr addrspace(8) %rsrc, ptr addrspace(3) %lptr, i32 8, i32 0, i32 0, i32 0, i32 0, i32 0)
  ret void
}
