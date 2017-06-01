; RUN: llc -mtriple=thumbv7 -mcpu=cortex-m0 < %s -disable-lsr | FileCheck %s
; FIXME: LSR mangles the last two testcases pretty badly. When this is fixed, remove
; the -disable-lsr above.

; CHECK-LABEL: @f
; CHECK: ldm {{r[0-9]}}!, {r{{[0-9]}}}
define i32 @f(i32* readonly %a, i32* readnone %b) {
  %1 = icmp eq i32* %a, %b
  br i1 %1, label %._crit_edge, label %.lr.ph

.lr.ph:                                           ; preds = %.lr.ph, %0
  %i.02 = phi i32 [ %3, %.lr.ph ], [ 0, %0 ]
  %.01 = phi i32* [ %4, %.lr.ph ], [ %a, %0 ]
  %2 = load i32, i32* %.01, align 4
  %3 = add nsw i32 %2, %i.02
  %4 = getelementptr inbounds i32, i32* %.01, i32 1
  %5 = icmp eq i32* %4, %b
  br i1 %5, label %._crit_edge, label %.lr.ph

._crit_edge:                                      ; preds = %.lr.ph, %0
  %i.0.lcssa = phi i32 [ 0, %0 ], [ %3, %.lr.ph ]
  ret i32 %i.0.lcssa
}

; CHECK-LABEL: @g
; CHECK-NOT: ldm
define i32 @g(i32* readonly %a, i32* readnone %b) {
  %1 = icmp eq i32* %a, %b
  br i1 %1, label %._crit_edge, label %.lr.ph

.lr.ph:                                           ; preds = %.lr.ph, %0
  %i.02 = phi i32 [ %3, %.lr.ph ], [ 0, %0 ]
  %.01 = phi i32* [ %4, %.lr.ph ], [ %a, %0 ]
  %2 = load i32, i32* %.01, align 4
  %3 = add nsw i32 %2, %i.02
  %4 = getelementptr inbounds i32, i32* %.01, i32 2
  %5 = icmp eq i32* %4, %b
  br i1 %5, label %._crit_edge, label %.lr.ph

._crit_edge:                                      ; preds = %.lr.ph, %0
  %i.0.lcssa = phi i32 [ 0, %0 ], [ %3, %.lr.ph ]
  ret i32 %i.0.lcssa
}

; CHECK-LABEL: @h
; CHECK: stm {{r[0-9]}}!, {r{{[0-9]}}}
define void @h(i32* %a, i32* readnone %b) {
  %1 = icmp eq i32* %a, %b
  br i1 %1, label %._crit_edge, label %.lr.ph

.lr.ph:                                           ; preds = %.lr.ph, %0
  %i.02 = phi i32 [ %2, %.lr.ph ], [ 0, %0 ]
  %.01 = phi i32* [ %3, %.lr.ph ], [ %a, %0 ]
  %2 = add nsw i32 %i.02, 1
  store i32 %i.02, i32* %.01, align 4
  %3 = getelementptr inbounds i32, i32* %.01, i32 1
  %4 = icmp eq i32* %3, %b
  br i1 %4, label %._crit_edge, label %.lr.ph

._crit_edge:                                      ; preds = %.lr.ph, %0
  ret void
}

; CHECK-LABEL: @j
; CHECK-NOT: stm
define void @j(i32* %a, i32* readnone %b) {
  %1 = icmp eq i32* %a, %b
  br i1 %1, label %._crit_edge, label %.lr.ph

.lr.ph:                                           ; preds = %.lr.ph, %0
  %i.02 = phi i32 [ %2, %.lr.ph ], [ 0, %0 ]
  %.01 = phi i32* [ %3, %.lr.ph ], [ %a, %0 ]
  %2 = add nsw i32 %i.02, 1
  store i32 %i.02, i32* %.01, align 4
  %3 = getelementptr inbounds i32, i32* %.01, i32 2
  %4 = icmp eq i32* %3, %b
  br i1 %4, label %._crit_edge, label %.lr.ph

._crit_edge:                                      ; preds = %.lr.ph, %0
  ret void
}
