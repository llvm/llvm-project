; RUN: opt -passes='print<loopnest>' %s 2>&1 | FileCheck %s
@global = external dso_local global [1000 x [1000 x i32]], align 16

; CHECK: IsPerfect=true, Depth=1, OutermostLoop: inner.header, Loops: ( inner.header )
; CHECK-NEXT: IsPerfect=true, Depth=2, OutermostLoop: outer.header, Loops: ( outer.header inner.header )

define void @foo1(i1 %cmp) {
entry:
  br i1 %cmp, label %bb1, label %bb1

bb1:                                              ; preds = %entry, %entry
  br i1 %cmp, label %outer.header.preheader, label %outer.header.preheader

outer.header.preheader:                           ; preds = %bb1, %bb1
  br label %outer.header

outer.header:                                     ; preds = %outer.header.preheader, %outer.latch
  %outer.iv = phi i64 [ %outer.iv.next, %outer.latch ], [ 0, %outer.header.preheader ]
  br i1 %cmp, label %inner.header.preheader, label %inner.header.preheader

inner.header.preheader:                           ; preds = %outer.header, %outer.header
  br label %inner.header

inner.header:                                     ; preds = %inner.header.preheader, %inner.header
  %inner.iv = phi i64 [ %inner.iv.next, %inner.header ], [ 5, %inner.header.preheader ]
  %ptr = getelementptr inbounds [1000 x [1000 x i32]], ptr @global, i64 0, i64 %inner.iv, i64 %outer.iv
  %lv = load i32, ptr %ptr, align 4
  %v = mul i32 %lv, 100
  store i32 %v, ptr %ptr, align 4
  %inner.iv.next = add nsw i64 %inner.iv, 1
  %cond1 = icmp eq i64 %inner.iv.next, 1000
  br i1 %cond1, label %outer.latch, label %inner.header

outer.latch:                                      ; preds = %inner.header
  %outer.iv.next = add nuw nsw i64 %outer.iv, 1
  %cond2 = icmp eq i64 %outer.iv.next, 1000
  br i1 %cond2, label %bb9, label %outer.header

bb9:                                              ; preds = %outer.latch
  br label %bb10

bb10:                                             ; preds = %bb9
  ret void
}
