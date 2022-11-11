; RUN: opt < %s -S -licm | FileCheck %s

; Check that the LICM pass does not operate on a loop which has the
; llvm.licm.disable metadata.
; CHECK-LABEL: @licm_disable
; CHECK: entry:
; CHECK-NOT: load
; CHECK: do.body:
; CHECK: load i64, ptr @in

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

@in = internal unnamed_addr global ptr null, align 8
@out = internal unnamed_addr global ptr null, align 8

define void @licm_disable(i32 %N) {
entry:
  br label %do.body

do.body:                                          ; preds = %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %do.body ]
  %v1 = load i64, ptr @in, align 8
  store i64 %v1, ptr @out, align 8
  %inc = add nsw i32 %i.0, 1
  %cmp = icmp slt i32 %inc, %N
  br i1 %cmp, label %do.body, label %do.end, !llvm.loop !1

do.end:                                           ; preds = %do.body
  ret void
}
!1 = !{!1, !2}
!2 = !{!"llvm.licm.disable"}
