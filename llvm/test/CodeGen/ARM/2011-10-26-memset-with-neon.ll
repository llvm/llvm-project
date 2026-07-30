; RUN: llc -mtriple=arm-eabi -mcpu=cortex-a8 %s -o - | FileCheck %s

; Trigger multiple NEON stores.
; CHECK: vst1.64
; CHECK: vst1.64
define void @f_0_40(ptr nocapture %c) nounwind optsize {
entry:
  call void @llvm.memset.p0.i64(ptr align 16 %c, i8 0, i64 40, i1 false)
  ret void
}

declare void @llvm.memset.p0.i64(ptr nocapture, i8, i64, i1) nounwind
