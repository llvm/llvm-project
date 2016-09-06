; RUN: llc < %s | FileCheck %s

target datalayout = "e-m:e-p:32:32-i1:8:32-i8:8:32-i16:16:32-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "thumbv7m--linux-gnu"

; CHECK-LABEL: f:
; CHECK: ldm	r{{[0-9]}}!, {r[[x:[0-9]]]}
; CHECK: add.w	r[[x]], r[[x]], #3
; CHECK: stm	r{{[0-9]}}!, {r[[x]]}
define void @f(i32 %n, i32* nocapture %a, i32* nocapture readonly %b) optsize minsize {
  %1 = icmp sgt i32 %n, 0
  br i1 %1, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %.lr.ph, %0
  %i.04 = phi i32 [ %6, %.lr.ph ], [ 0, %0 ]
  %.03 = phi i32* [ %2, %.lr.ph ], [ %b, %0 ]
  %.012 = phi i32* [ %5, %.lr.ph ], [ %a, %0 ]
  %2 = getelementptr inbounds i32, i32* %.03, i32 1
  %3 = load i32, i32* %.03, align 4
  %4 = add nsw i32 %3, 3
  %5 = getelementptr inbounds i32, i32* %.012, i32 1
  store i32 %4, i32* %.012, align 4
  %6 = add nsw i32 %i.04, 1
  %exitcond = icmp eq i32 %6, %n
  br i1 %exitcond, label %._crit_edge, label %.lr.ph

._crit_edge:                                      ; preds = %.lr.ph, %0
  ret void
}

; CHECK-LABEL: f_nominsize:
; CHECK-NOT: ldm
define void @f_nominsize(i32 %n, i32* nocapture %a, i32* nocapture readonly %b) optsize {
  %1 = icmp sgt i32 %n, 0
  br i1 %1, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %.lr.ph, %0
  %i.04 = phi i32 [ %6, %.lr.ph ], [ 0, %0 ]
  %.03 = phi i32* [ %2, %.lr.ph ], [ %b, %0 ]
  %.012 = phi i32* [ %5, %.lr.ph ], [ %a, %0 ]
  %2 = getelementptr inbounds i32, i32* %.03, i32 1
  %3 = load i32, i32* %.03, align 4
  %4 = add nsw i32 %3, 3
  %5 = getelementptr inbounds i32, i32* %.012, i32 1
  store i32 %4, i32* %.012, align 4
  %6 = add nsw i32 %i.04, 1
  %exitcond = icmp eq i32 %6, %n
  br i1 %exitcond, label %._crit_edge, label %.lr.ph

._crit_edge:                                      ; preds = %.lr.ph, %0
  ret void
}
