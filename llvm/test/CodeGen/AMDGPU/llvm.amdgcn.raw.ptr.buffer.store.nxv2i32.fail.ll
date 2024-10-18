; RUN: not --crash llc -mtriple=amdgcn -mcpu=gfx900 < %s
; RUN: not --crash llc -global-isel=1 -mtriple=amdgcn -mcpu=gfx900 < %s
define void @buffer_store_nxv2i32(ptr addrspace(8) inreg %rsrc, i32 %offset) {
  call void @llvm.amdgcn.raw.ptr.buffer.store.nxv2i32(<vscale x 2 x i32> poison, ptr addrspace(8) %rsrc, i32 %offset, i32 0, i32 0)
  ret void
}
