; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_80 -mattr=+ptx81 | FileCheck %s

declare { float, float, float, float } @llvm.nvvm.mma.m16n8k4.row.col.tf32(i32, i32, i32, float, float, float, float) #1

declare noundef i32 @llvm.nvvm.read.ptx.sreg.laneid() #0

; COM: llvm.nvvm.mma should not sink to the next block and gets reordered to be after laneid check.
; CHECK-LABEL: no_reorder_mma_and_laneid_check
define dso_local void @no_reorder_mma_and_laneid_check(ptr %arg, ptr %arg1) {
bb:
  ; CHECK: mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32
  ; CHECK: laneid
  %i = tail call { float, float, float, float } @llvm.nvvm.mma.m16n8k4.row.col.tf32(i32 10, i32 10, i32 8, float 0.0, float 0.0, float 0.0, float 0.0)
  %i3 = tail call i32 @llvm.nvvm.read.ptx.sreg.laneid()
  %i4 = icmp eq i32 %i3, 0
  br i1 %i4, label %bb5, label %bb8

bb5:                                              ; preds = %bb
  %i6 = extractvalue { float, float, float, float } %i, 0
  %i7 = getelementptr float, ptr %arg, i64 0
  store float %i6, ptr %i7, align 4
  br label %bb8

bb8:                                              ; preds = %bb5, %bb
  ret void
}
