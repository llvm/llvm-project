; RUN: llc -global-isel=0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 < %s | FileCheck %s

; Test that DAGCombiner::reassociationCanBreakAddressingModePattern does not
; crash when a MemSDNode user has multiple memory operands (e.g.
; buffer_load_lds which reads from a buffer and writes to LDS).

@global_smem = external addrspace(3) global [0 x i8], align 16

declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1), i16, i64, i32)
declare void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8), ptr addrspace(3) nocapture, i32, i32, i32, i32, i32)
declare i32 @llvm.amdgcn.workitem.id.x()

define amdgpu_kernel void @triton_mm_minimal(ptr addrspace(1) inreg %ptr) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  ; Create a pattern that will be reassociated: (add (add base, 1024), 32)
  ; where base comes from mul, creating nested adds
  %base = mul i32 %tid, 1536
  %add1 = add i32 %base, 1024
  %offset1 = add i32 %add1, 32
  %offset2 = add i32 %add1, 33
  %shl1 = shl i32 %offset1, 1
  %shl2 = shl i32 %offset2, 1
  %rsrc = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %ptr, i16 0, i64 2147483646, i32 159744)
  %lds0 = getelementptr inbounds i8, ptr addrspace(3) @global_smem, i32 0
  %lds1 = getelementptr inbounds i8, ptr addrspace(3) @global_smem, i32 1056
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %rsrc, ptr addrspace(3) %lds0, i32 16, i32 %shl1, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %rsrc, ptr addrspace(3) %lds1, i32 16, i32 %shl2, i32 0, i32 0, i32 0)
  ret void
}
