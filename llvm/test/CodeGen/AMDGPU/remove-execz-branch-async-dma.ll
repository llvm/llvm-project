; RUN: llc -mtriple=amdgcn -mcpu=gfx1250 < %s | FileCheck %s

@lds = external addrspace(3) global [0 x i8], align 16

declare void @llvm.amdgcn.global.load.async.to.lds.b128(ptr addrspace(1), ptr addrspace(3), i32 immarg, i32 immarg)
declare void @llvm.amdgcn.s.wait.asynccnt(i16 immarg)
declare i32 @llvm.amdgcn.workitem.id.x()

; The cost model treats async DMA at its 1-cycle issue cost rather than the
; scheduling model's full memory latency. When the guarded block contains
; only cheap ALU and the async load (no waitcnts), the branch is removed.
;
; CHECK-LABEL: async_load_no_waitcnt:
; CHECK-NOT: s_cbranch_execz
; CHECK: global_load_async_to_lds_b128
; CHECK: s_or_b32 exec_lo
define amdgpu_ps void @async_load_no_waitcnt(
    ptr addrspace(1) inreg %src,
    i32 %bound
) {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %lds_off = shl nuw nsw i32 %tid, 4
  %lds_ptr = getelementptr inbounds i8, ptr addrspace(3) @lds, i32 %lds_off
  %off = sext i32 %tid to i64
  %gep = getelementptr i8, ptr addrspace(1) %src, i64 %off
  %cmp = icmp slt i32 %tid, %bound
  br i1 %cmp, label %do_load, label %skip

do_load:
  tail call void @llvm.amdgcn.global.load.async.to.lds.b128(
      ptr addrspace(1) %gep,
      ptr addrspace(3) %lds_ptr,
      i32 0, i32 0)
  br label %join

skip:
  br label %join

join:
  tail call void @llvm.amdgcn.s.wait.asynccnt(i16 0)
  ret void
}
