; RUN: llc -march=arm -mcpu=cortex-a8 < %s | FileCheck %s

; Should trigger a NEON store.
; CHECK: vstr
define void @f_0_12(i8* nocapture %c) nounwind optsize {
entry:
  call void @llvm.memset.p0i8.i64(i8* %c, i8 0, i64 12, i32 8, i1 false)
  ret void
}

; Trigger multiple NEON stores.
; CHECK:      vst1.64
; CHECK-NEXT: vst1.64
define void @f_0_40(i8* nocapture %c) nounwind optsize {
entry:
  call void @llvm.memset.p0i8.i64(i8* %c, i8 0, i64 40, i32 16, i1 false)
  ret void
}

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1) nounwind
