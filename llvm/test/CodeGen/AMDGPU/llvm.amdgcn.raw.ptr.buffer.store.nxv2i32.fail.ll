; Note: The exact error messages aren't important here, but are included to catch
; anything changing.
; RUN: not --crash llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx900  -filetype=null < %s 2>&1 | FileCheck %s --check-prefix=SDAG
; SDAG: LLVM ERROR: Scalarization of scalable vectors is not supported.
; RUN: not --crash llc -global-isel=1 -mtriple=amdgcn -mcpu=gfx900  -filetype=null < %s 2>&1 | FileCheck %s --check-prefix=GISEL
; GISEL: LLVM ERROR: Cannot implicitly convert a scalable size to a fixed-width size in `TypeSize::operator ScalarTy()`

define void @buffer_store_nxv2i32(ptr addrspace(8) inreg %rsrc, i32 %offset) {
  call void @llvm.amdgcn.raw.ptr.buffer.store.nxv2i32(<vscale x 2 x i32> poison, ptr addrspace(8) %rsrc, i32 %offset, i32 0, i32 0)
  ret void
}
