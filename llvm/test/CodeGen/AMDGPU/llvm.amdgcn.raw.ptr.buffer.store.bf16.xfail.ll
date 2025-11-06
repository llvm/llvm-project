; RUN: not --crash llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=null %s 2>&1 | FileCheck %s

; FIXME: This should be handled

; CHECK: LLVM ERROR: Do not know how to widen this operator's operand!


define void @buffer_store_v6bf16(ptr addrspace(8) inreg %rsrc, <6 x bfloat> %data, i32 %offset) {
  call void @llvm.amdgcn.raw.ptr.buffer.store.v6bf16(<6 x bfloat> %data, ptr addrspace(8) %rsrc, i32 %offset, i32 0, i32 0)
  ret void
}
