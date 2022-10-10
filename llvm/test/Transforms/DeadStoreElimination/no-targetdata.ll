; RUN: opt -dse -S < %s | FileCheck %s

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture, i64, i1) nounwind

define void @fn(ptr nocapture %buf) #0 {
entry:

; We would not eliminate the first memcpy with data layout, and we should not
; eliminate it without data layout.
; CHECK-LABEL: @fn
; CHECK: tail call void @llvm.memcpy.p0.p0.i64
; CHECK: tail call void @llvm.memcpy.p0.p0.i64
; CHECK: ret void

  %arrayidx = getelementptr i8, ptr %buf, i64 18
  tail call void @llvm.memcpy.p0.p0.i64(ptr %arrayidx, ptr %buf, i64 18, i1 false)
  store i8 1, ptr %arrayidx, align 1
  tail call void @llvm.memcpy.p0.p0.i64(ptr %buf, ptr %arrayidx, i64 18, i1 false)
  ret void
}

