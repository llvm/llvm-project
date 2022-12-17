; RUN: opt < %s -passes=loop-interchange -cache-line-size=64 -loop-interchange-threshold=-100 -verify-loop-lcssa -S | FileCheck %s

; Test case for PR41725. The induction variables in the latches escape the
; loops and we must move some PHIs around.

@a = common dso_local global i64 0, align 4
@b = common dso_local global i64 0, align 4
@c = common dso_local global [10 x [10 x i32 ]] zeroinitializer, align 16


define void @test_lcssa_indvars1()  {
; CHECK-LABEL: @test_lcssa_indvars1()
; CHECK-LABEL: inner.body:
; CHECK-NEXT:    %iv.inner = phi i64 [ %[[IVNEXT:[0-9]+]], %inner.body.split ], [ 5, %inner.body.preheader ]

; CHECK-LABEL: inner.body.split:
; CHECK-NEXT:    %0 = phi i64 [ %iv.outer.next, %outer.latch ]
; CHECK-NEXT:    %[[IVNEXT]] = add nsw i64 %iv.inner, -1
; CHECK-NEXT:    %[[COND:[0-9]+]] = icmp eq i64 %iv.inner, 0
; CHECK-NEXT:    br i1 %[[COND]], label %exit, label %inner.body

; CHECK-LABEL: exit:
; CHECK-NEXT:    %v4.lcssa = phi i64 [ %0, %inner.body.split ]
; CHECK-NEXT:    %v8.lcssa.lcssa = phi i64 [ %[[IVNEXT]], %inner.body.split ]
; CHECK-NEXT:    store i64 %v8.lcssa.lcssa, i64* @b, align 4
; CHECK-NEXT:    store i64 %v4.lcssa, i64* @a, align 4

entry:
  br label %outer.header

outer.header:                                     ; preds = %outer.latch, %entry
  %iv.outer = phi i64 [ 0, %entry ], [ %iv.outer.next, %outer.latch ]
  br label %inner.body

inner.body:                                       ; preds = %inner.body, %outer.header
  %iv.inner = phi i64 [ 5, %outer.header ], [ %iv.inner.next, %inner.body ]
  %v7 = getelementptr inbounds [10 x [10 x i32]], [10 x [10 x i32]]* @c, i64 0, i64 %iv.inner, i64 %iv.outer
  store i32 0, i32* %v7, align 4
  %iv.inner.next = add nsw i64 %iv.inner, -1
  %v9 = icmp eq i64 %iv.inner, 0
  br i1 %v9, label %outer.latch, label %inner.body

outer.latch:                                      ; preds = %inner.body
  %v8.lcssa = phi i64 [ %iv.inner.next, %inner.body ]
  %iv.outer.next = add nuw nsw i64 %iv.outer, 1
  %v5 = icmp ult i64 %iv.outer, 2
  br i1 %v5, label %outer.header, label %exit

exit:                                             ; preds = %outer.latch
  %v4.lcssa = phi i64 [ %iv.outer.next, %outer.latch ]
  %v8.lcssa.lcssa = phi i64 [ %v8.lcssa, %outer.latch ]
  store i64 %v8.lcssa.lcssa, i64* @b, align 4
  store i64 %v4.lcssa, i64* @a, align 4
  ret void
}


define void @test_lcssa_indvars2()  {
; CHECK-LABEL: @test_lcssa_indvars2()
; CHECK-LABEL: inner.body:
; CHECK-NEXT:    %iv.inner = phi i64 [ %[[IVNEXT:[0-9]+]], %inner.body.split ], [ 5, %inner.body.preheader ]

; CHECK-LABEL: inner.body.split:
; CHECK-NEXT:    %0 = phi i64 [ %iv.outer, %outer.latch ]
; CHECK-NEXT:    %[[IVNEXT]] = add nsw i64 %iv.inner, -1
; CHECK-NEXT:    %[[COND:[0-9]+]] = icmp eq i64 %[[IVNEXT]], 0
; CHECK-NEXT:    br i1 %[[COND]], label %exit, label %inner.body

; CHECK-LABEL: exit:
; CHECK-NEXT:    %v4.lcssa = phi i64 [ %0, %inner.body.split ]
; CHECK-NEXT:    %v8.lcssa.lcssa = phi i64 [ %iv.inner, %inner.body.split ]
; CHECK-NEXT:    store i64 %v8.lcssa.lcssa, i64* @b, align 4
; CHECK-NEXT:    store i64 %v4.lcssa, i64* @a, align 4

entry:
  br label %outer.header

outer.header:                                     ; preds = %outer.latch, %entry
  %iv.outer = phi i64 [ 0, %entry ], [ %iv.outer.next, %outer.latch ]
  br label %inner.body

inner.body:                                       ; preds = %inner.body, %outer.header
  %iv.inner = phi i64 [ 5, %outer.header ], [ %iv.inner.next, %inner.body ]
  %v7 = getelementptr inbounds [10 x [10 x i32]], [10 x [10 x i32]]* @c, i64 0, i64 %iv.inner, i64 %iv.outer
  store i32 0, i32* %v7, align 4
  %iv.inner.next = add nsw i64 %iv.inner, -1
  %v9 = icmp eq i64 %iv.inner.next, 0
  br i1 %v9, label %outer.latch, label %inner.body

outer.latch:                                      ; preds = %inner.body
  %v8.lcssa = phi i64 [ %iv.inner, %inner.body ]
  %iv.outer.next = add nuw nsw i64 %iv.outer, 1
  %v5 = icmp ult i64 %iv.outer.next, 2
  br i1 %v5, label %outer.header, label %exit

exit:                                             ; preds = %outer.latch
  %v4.lcssa = phi i64 [ %iv.outer, %outer.latch ]
  %v8.lcssa.lcssa = phi i64 [ %v8.lcssa, %outer.latch ]
  store i64 %v8.lcssa.lcssa, i64* @b, align 4
  store i64 %v4.lcssa, i64* @a, align 4
  ret void
}

define void @test_lcssa_indvars3()  {
; CHECK-LABEL: @test_lcssa_indvars3()
; CHECK-LABEL: inner.body:
; CHECK-NEXT:    %iv.inner = phi i64 [ %[[IVNEXT:[0-9]+]], %inner.body.split ], [ 5, %inner.body.preheader ]

; CHECK-LABEL: inner.body.split:
; CHECK-NEXT:    %0 = phi i64 [ %iv.outer.next, %outer.latch ]
; CHECK-NEXT:    %[[IVNEXT]] = add nsw i64 %iv.inner, -1
; CHECK-NEXT:    %[[COND:[0-9]+]] = icmp eq i64 %iv.inner, 0
; CHECK-NEXT:    br i1 %[[COND]], label %exit, label %inner.body

; CHECK-LABEL: exit:
; CHECK-NEXT:    %v4.lcssa = phi i64 [ %0, %inner.body.split ]
; CHECK-NEXT:    %v8.lcssa.lcssa = phi i64 [ %[[IVNEXT]], %inner.body.split ]
; CHECK-NEXT:    %v8.lcssa.lcssa.2 = phi i64 [ %[[IVNEXT]], %inner.body.split ]
; CHECK-NEXT:    %r1 = add i64 %v8.lcssa.lcssa, %v8.lcssa.lcssa.2
; CHECK-NEXT:    store i64 %r1, i64* @b, align 4
; CHECK-NEXT:    store i64 %v4.lcssa, i64* @a, align 4


entry:
  br label %outer.header

outer.header:                                     ; preds = %outer.latch, %entry
  %iv.outer = phi i64 [ 0, %entry ], [ %iv.outer.next, %outer.latch ]
  br label %inner.body

inner.body:                                       ; preds = %inner.body, %outer.header
  %iv.inner = phi i64 [ 5, %outer.header ], [ %iv.inner.next, %inner.body ]
  %v7 = getelementptr inbounds [10 x [10 x i32]], [10 x [10 x i32]]* @c, i64 0, i64 %iv.inner, i64 %iv.outer
  store i32 0, i32* %v7, align 4
  %iv.inner.next = add nsw i64 %iv.inner, -1
  %v9 = icmp eq i64 %iv.inner, 0
  br i1 %v9, label %outer.latch, label %inner.body

outer.latch:                                      ; preds = %inner.body
  %v8.lcssa = phi i64 [ %iv.inner.next, %inner.body ]
  ;%const.lcssa = phi i64 [ 111, %inner.body ]
  %iv.outer.next = add nuw nsw i64 %iv.outer, 1
  %v5 = icmp ult i64 %iv.outer, 2
  br i1 %v5, label %outer.header, label %exit

exit:                                             ; preds = %outer.latch
  %v4.lcssa = phi i64 [ %iv.outer.next, %outer.latch ]
  %v8.lcssa.lcssa = phi i64 [ %v8.lcssa, %outer.latch ]
  %v8.lcssa.lcssa.2 = phi i64 [ %v8.lcssa, %outer.latch ]
  %r1 = add i64 %v8.lcssa.lcssa, %v8.lcssa.lcssa.2
  store i64 %r1, i64* @b, align 4
  store i64 %v4.lcssa, i64* @a, align 4
  ret void
}


; Make sure we do not crash for loops without reachable exits.
define void @no_reachable_exits() {
; Check we interchanged.
; CHECK-LABEL: @no_reachable_exits() {
; CHECK-NEXT:  bb:
; CHECK-NEXT:    br label %inner.ph
; CHECK-LABEL: outer.ph:
; CHECK-NEXT:    br label %outer.header
; CHECK-LABEL: inner.ph:
; CHECK-NEXT:    br label %inner.body
; CHECK-LABEL: inner.body:
; CHECK-NEXT:    %tmp31 = phi i32 [ 0, %inner.ph ], [ %[[IVNEXT:[0-9]]], %inner.body.split ]
; CHECK-NEXT:    br label %outer.ph
; CHECK-LABEL: inner.body.split:
; CHECK-NEXT:    %[[IVNEXT]] = add nsw i32 %tmp31, 1
; CHECK-NEXT:    br i1 false, label %inner.body, label %exit


bb:
  br label %outer.ph

outer.ph:                              ; preds = %bb
  br label %outer.header

outer.header:                                    ; preds = %outer.ph, %outer.latch
  %tmp2 = phi i32 [ 0, %outer.ph ], [ %tmp8, %outer.latch ]
  br i1 undef, label %inner.ph, label %outer.latch

inner.ph:                                        ; preds = %outer.header
  br label %inner.body

inner.body:                                              ; preds = %inner.ph, %inner.body
  %tmp31 = phi i32 [ 0, %inner.ph ], [ %tmp6, %inner.body]
  %tmp5 = load i32*, i32** undef, align 8
  %tmp6 = add nsw i32 %tmp31, 1
  br i1 undef, label %inner.body, label %outer.latch

outer.latch:                                              ; preds = %inner.body, %outer.header
  %tmp8 = add nsw i32 %tmp2, 1
  br i1 undef, label %outer.header, label %exit

exit:                                              ; preds = %outer.latch
  unreachable
}
