; RUN: not --crash llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=null %s 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: Do not know how to widen the result of this operator!

define <6 x bfloat> @raw_ptr_buffer_load_v6bf16(ptr addrspace(8) inreg %rsrc) {
  %val = call <6 x bfloat> @llvm.amdgcn.raw.ptr.buffer.load.v6bf16(ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0)
  ret <6 x bfloat> %val
}
