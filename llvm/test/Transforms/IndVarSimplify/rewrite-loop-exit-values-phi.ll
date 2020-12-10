; RUN: opt -indvars -S %s -o - | FileCheck %s

; When bailing out in rewriteLoopExitValues() you would be left with a PHI node
; that was not deleted, and the IndVar pass would return an incorrect modified
; status. This was caught by the expensive check introduced in D86589.

; CHECK-LABEL: header:
; CHECK-NEXT: %idx = phi i64 [ %idx.next, %latch ], [ undef, %entry ]
; CHECK-NEXT: %cond = icmp sgt i64 %n, %idx
; CHECK-NEXT: br i1 %cond, label %end, label %inner.preheader

; CHECK-LABEL: latch:
; CHECK-NEXT: %idx.next = add nsw i64 %idx, -1
; CHECK-NEXT: br label %header

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@ptr = external global i64

define dso_local void @hoge() local_unnamed_addr {
entry:                                            ; preds = %entry
  %n = sdiv exact i64 undef, 40
  br label %header

header:                                           ; preds = %latch, %entry
  %idx = phi i64 [ %idx.next, %latch ], [ undef, %entry ]
  %cond = icmp sgt i64 %n, %idx
  br i1 %cond, label %end, label %inner

inner:                                            ; preds = %inner, %header
  %i = phi i64 [ %i.next, %inner ], [ 0, %header ]
  %j = phi i64 [ %j.next, %inner ], [ %n, %header ]
  %i.next = add nsw i64 %i, 1
  %j.next = add nsw i64 %j, 1
  store i64 undef, i64* @ptr
  %cond1 = icmp slt i64 %j, %idx
  br i1 %cond1, label %inner, label %inner_exit

inner_exit:                                       ; preds = %inner
  %indvar = phi i64 [ %i.next, %inner ]
  %indvar_use = add i64 %indvar, 1
  br label %latch

latch:                                            ; preds = %inner_exit
  %idx.next = add nsw i64 %idx, -1
  br label %header

end:                                              ; preds = %header
  ret void
}
