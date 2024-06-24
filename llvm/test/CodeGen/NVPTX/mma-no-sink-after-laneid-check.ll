; RUN: llc < %s -march=nvptx64 -mcpu=sm_80 -mattr=+ptx81 | FileCheck %s

declare noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

declare noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #1

declare noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

declare { float, float, float, float } @llvm.nvvm.mma.m16n8k4.row.col.tf32(i32, i32, i32, float, float, float, float) #2

declare noundef i32 @llvm.nvvm.read.ptx.sreg.laneid() #1

; COM: llvm.nvvm.mma should not sink to the next block and gets reordered to be after laneid check.
; CHECK-LABEL: no_reorder_mma_and_laneid_check
define dso_local void @no_reorder_mma_and_laneid_check(ptr %0, ptr %1, i64 %2) #0 {
3:
  ; CHECK: mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32
  ; CHECK: laneid
  %4 = tail call { float, float, float, float } @llvm.nvvm.mma.m16n8k4.row.col.tf32(i32 1065353216, i32 1065353216, i32 9, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00)
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.laneid()
  %6 = icmp eq i32 %5, 0
  br i1 %6, label %7, label %10

7:                                               ; preds = %3
  %8 = extractvalue { float, float, float, float } %4, 0
  %9 = getelementptr float, ptr %0, i64 0
  store float %8, ptr %9, align 4
  br label %10

10:                                               ; preds = %3, %7
  ret void
}
