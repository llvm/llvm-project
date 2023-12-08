; RUN: opt -passes=loop-interchange -cache-line-size=64 -verify-dom-info -verify-loop-info -verify-loop-lcssa -S %s | FileCheck %s

@b = global [3 x [5 x [8 x i16]]] [[5 x [8 x i16]] zeroinitializer, [5 x [8 x i16]] [[8 x i16] zeroinitializer, [8 x i16] [i16 0, i16 0, i16 0, i16 6, i16 1, i16 6, i16 0, i16 0], [8 x i16] zeroinitializer, [8 x i16] zeroinitializer, [8 x i16] zeroinitializer], [5 x [8 x i16]] zeroinitializer], align 2
@a = common global i32 0, align 4
@d = common dso_local local_unnamed_addr global [1 x [6 x i32]] zeroinitializer, align 4


;  Doubly nested loop
;; C test case:
;; int a;
;; short b[3][5][8] = {{}, {{}, 0, 0, 0, 6, 1, 6}};
;; void test1() {
;;   int c = 0, d;
;;   for (; c <= 2; c++) {
;;     if (c)
;;       continue;
;;     d = 0;
;;     for (; d <= 2; d++)
;;       a |= b[d][d][c + 5];
;;   }
;; }

define void @test1() {
;CHECK-LABEL: @test1(
;CHECK:          entry:
;CHECK-NEXT:       br label [[FOR_COND1_PREHEADER:%.*]]
;CHECK:          for.body.preheader:
;CHECK-NEXT:       br label  [[FOR_BODY:%.*]]
;CHECK:          for.body:
;CHECK-NEXT:       [[INDVARS_IV22:%.*]] = phi i64 [ [[INDVARS_IV_NEXT23:%.*]], [[FOR_INC8:%.*]] ], [ 0, [[FOR_BODY_PREHEADER:%.*]] ]
;CHECK-NEXT:       [[TOBOOL:%.*]] = icmp eq i64 [[INDVARS_IV22:%.*]], 0
;CHECK-NEXT:       br i1 [[TOBOOL]], label [[FOR_BODY3_SPLIT1:%.*]], label [[FOR_BODY3_SPLIT:%.*]]
;CHECK:          for.cond1.preheader:
;CHECK-NEXT:       br label [[FOR_BODY3:%.*]]
;CHECK:          for.body3:
;CHECK-NEXT:       [[INDVARS_IV:%.*]] = phi i64 [ 0, [[FOR_COND1_PREHEADER]] ], [ %3, [[FOR_BODY3_SPLIT]] ]
;CHECK-NEXT:        br label [[FOR_BODY_PREHEADER]]
;CHECK:          for.body3.split1:
;CHECK-NEXT:       [[TMP0:%.*]] = add nuw nsw i64 [[INDVARS_IV22]], 5
;CHECK-NEXT:       [[ARRAYIDX7:%.*]] = getelementptr inbounds [3 x [5 x [8 x i16]]], ptr @b, i64 0, i64 [[INDVARS_IV]], i64 [[INDVARS_IV]], i64 [[TMP0]]
;CHECK-NEXT:       [[TMP1:%.*]] = load i16, ptr [[ARRAYIDX7]]
;CHECK-NEXT:       [[CONV:%.*]] = sext i16 [[TMP1]] to i32
;CHECK-NEXT:       [[TMP2:%.*]] = load i32, ptr @a
;CHECK-NEXT:       [[TMP_OR:%.*]] = or i32 [[TMP2]], [[CONV]]
;CHECK-NEXT:       store i32 [[TMP_OR]], ptr @a
;CHECK-NEXT:       [[INDVARS_IV_NEXT:%.*]] = add nuw nsw i64 [[INDVARS_IV]], 1
;CHECK-NEXT:       [[EXITCOND:%.*]] = icmp ne i64 [[INDVARS_IV_NEXT]], 3
;CHECK-NEXT:       br label [[FOR_INC8_LOOPEXIT:%.*]]
;CHECK:          for.body3.split:
;CHECK-NEXT:       [[TMP3:%.*]] = add nuw nsw i64 [[INDVARS_IV]], 1
;CHECK-NEXT:       [[TMP4:%.*]] = icmp ne i64 [[TMP3]], 3
;CHECK-NEXT:       br i1 %4, label [[FOR_BODY3]], label [[FOR_END10:%.*]]
;CHECK:          for.inc8.loopexit:
;CHECK-NEXT:       br label [[FOR_INC8]]
;CHECK:          for.inc8:
;CHECK-NEXT:       [[INDVARS_IV_NEXT23]] = add nuw nsw i64 [[INDVARS_IV22]], 1
;CHECK-NEXT:       [[EXITCOND25:%.*]] = icmp ne i64 [[INDVARS_IV_NEXT23]], 3
;CHECK-NEXT:       br i1 [[EXITCOND25]], label [[FOR_BODY]], label [[FOR_BODY3_SPLIT]]
;CHECK:         for.end10:
;CHECK-NEXT:       [[TMP5:%.*]] = load i32, ptr @a
;CHECK-NEXT:       ret void

entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.inc8
  %indvars.iv22 = phi i64 [ 0, %entry ], [ %indvars.iv.next23, %for.inc8 ]
  %tobool = icmp eq i64 %indvars.iv22, 0
  br i1 %tobool, label %for.cond1.preheader, label %for.inc8

for.cond1.preheader:                              ; preds = %for.body
  br label %for.body3

for.body3:                                        ; preds = %for.cond1.preheader, %for.body3
  %indvars.iv = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next, %for.body3 ]
  %0 = add nuw nsw i64 %indvars.iv22, 5
  %arrayidx7 = getelementptr inbounds [3 x [5 x [8 x i16]]], ptr @b, i64 0, i64 %indvars.iv, i64 %indvars.iv, i64 %0
  %1 = load i16, ptr %arrayidx7
  %conv = sext i16 %1 to i32
  %2 = load i32, ptr @a
  %or = or i32 %2, %conv
  store i32 %or, ptr @a
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 3
  br i1 %exitcond, label %for.body3, label %for.inc8.loopexit

for.inc8.loopexit:                                ; preds = %for.body3
  br label %for.inc8

for.inc8:                                         ; preds = %for.inc8.loopexit, %for.body
  %indvars.iv.next23 = add nuw nsw i64 %indvars.iv22, 1
  %exitcond25 = icmp ne i64 %indvars.iv.next23, 3
  br i1 %exitcond25, label %for.body, label %for.end10

for.end10:                                        ; preds = %for.inc8
  %3 = load i32, ptr @a
  ret void
}

; Triply nested loop
; The innermost and the middle loop are interchanged.
; C test case:
;; a;
;; d[][6];
;; void test2() {
;;   int g = 10;
;;   for (; g; g = g - 5) {
;;     short c = 4;
;;     for (; c; c--) {
;;       int i = 4;
;;       for (; i; i--) {
;;         if (a)
;;           break;
;;         d[i][c] = 0;
;;       }
;;     }
;;   }
;; }

define void @test2() {
; CHECK-LABEL: @test2(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[OUTERMOST_HEADER:%.*]]
; CHECK:       outermost.header:
; CHECK-NEXT:    [[INDVAR_OUTERMOST:%.*]] = phi i32 [ 10, [[ENTRY:%.*]] ], [ [[INDVAR_OUTERMOST_NEXT:%.*]], [[OUTERMOST_LATCH:%.*]] ]
; CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr @a, align 4
; CHECK-NEXT:    [[TOBOOL71_I:%.*]] = icmp eq i32 [[TMP0]], 0
; CHECK-NEXT:    br label [[INNERMOST_PREHEADER:%.*]]
; CHECK:       middle.header.preheader:
; CHECK-NEXT:    br label [[MIDDLE_HEADER:%.*]]
; CHECK:       middle.header:
; CHECK-NEXT:    [[INDVAR_MIDDLE:%.*]] = phi i64 [ [[INDVAR_MIDDLE_NEXT:%.*]], [[MIDDLE_LATCH:%.*]] ], [ 4, [[MIDDLE_HEADER_PREHEADER:%.*]] ]
; CHECK-NEXT:    br i1 [[TOBOOL71_I]], label [[INNERMOST_BODY_SPLIT1:%.*]], label [[INNERMOST_BODY_SPLIT:%.*]]
; CHECK:       innermost.preheader:
; CHECK-NEXT:    br label [[INNERMOST_BODY:%.*]]
; CHECK:       innermost.body:
; CHECK-NEXT:    [[INDVAR_INNERMOST:%.*]] = phi i64 [ [[TMP1:%.*]], [[INNERMOST_BODY_SPLIT]] ], [ 4, [[INNERMOST_PREHEADER]] ]
; CHECK-NEXT:    br label [[MIDDLE_HEADER_PREHEADER]]
; CHECK:       innermost.body.split1:
; CHECK-NEXT:    [[ARRAYIDX9_I:%.*]] = getelementptr inbounds [1 x [6 x i32]], ptr @d, i64 0, i64 [[INDVAR_INNERMOST]], i64 [[INDVAR_MIDDLE]]
; CHECK-NEXT:    store i32 0, ptr [[ARRAYIDX9_I]], align 4
; CHECK-NEXT:    [[INDVAR_INNERMOST_NEXT:%.*]] = add nsw i64 [[INDVAR_INNERMOST]], -1
; CHECK-NEXT:    [[TOBOOL5_I:%.*]] = icmp eq i64 [[INDVAR_INNERMOST_NEXT]], 0
; CHECK-NEXT:    br label [[MIDDLE_LATCH_LOOPEXIT:%.*]]
; CHECK:       innermost.body.split:
; CHECK-NEXT:    [[TMP1]] = add nsw i64 [[INDVAR_INNERMOST]], -1
; CHECK-NEXT:    [[TMP2:%.*]] = icmp eq i64 [[TMP1]], 0
; CHECK-NEXT:    br i1 [[TMP2]], label [[OUTERMOST_LATCH]], label [[INNERMOST_BODY]]
; CHECK:       innermost.loopexit:
; CHECK-NEXT:    br label [[MIDDLE_LATCH]]
; CHECK:       middle.latch:
; CHECK-NEXT:    [[INDVAR_MIDDLE_NEXT]] = add nsw i64 [[INDVAR_MIDDLE]], -1
; CHECK-NEXT:    [[TOBOOL2_I:%.*]] = icmp eq i64 [[INDVAR_MIDDLE_NEXT]], 0
; CHECK-NEXT:    br i1 [[TOBOOL2_I]], label [[INNERMOST_BODY_SPLIT]], label [[MIDDLE_HEADER]]
; CHECK:       outermost.latch:
; CHECK-NEXT:    [[INDVAR_OUTERMOST_NEXT]] = add nsw i32 [[INDVAR_OUTERMOST]], -5
; CHECK-NEXT:    [[TOBOOL_I:%.*]] = icmp eq i32 [[INDVAR_OUTERMOST_NEXT]], 0
; CHECK-NEXT:    br i1 [[TOBOOL_I]], label [[OUTERMOST_EXIT:%.*]], label [[OUTERMOST_HEADER]]
; CHECK:       outermost.exit:
; CHECK-NEXT:    ret void
;

entry:
  br label %outermost.header

outermost.header:                      ; preds = %outermost.latch, %entry
  %indvar.outermost = phi i32 [ 10, %entry ], [ %indvar.outermost.next, %outermost.latch ]
  %0 = load i32, ptr @a, align 4
  %tobool71.i = icmp eq i32 %0, 0
  br label %middle.header

middle.header:                            ; preds = %middle.latch, %outermost.header
  %indvar.middle = phi i64 [ 4, %outermost.header ], [ %indvar.middle.next, %middle.latch ]
  br i1 %tobool71.i, label %innermost.preheader, label %middle.latch

innermost.preheader:                               ; preds = %middle.header
  br label %innermost.body

innermost.body:                                         ; preds = %innermost.preheader, %innermost.body
  %indvar.innermost = phi i64 [ %indvar.innermost.next, %innermost.body ], [ 4, %innermost.preheader ]
  %arrayidx9.i = getelementptr inbounds [1 x [6 x i32]], ptr @d, i64 0, i64 %indvar.innermost, i64 %indvar.middle
  store i32 0, ptr %arrayidx9.i, align 4
  %indvar.innermost.next = add nsw i64 %indvar.innermost, -1
  %tobool5.i = icmp eq i64 %indvar.innermost.next, 0
  br i1 %tobool5.i, label %innermost.loopexit, label %innermost.body

innermost.loopexit:                             ; preds = %innermost.body
  br label %middle.latch

middle.latch:                                      ; preds = %middle.latch.loopexit, %middle.header
  %indvar.middle.next = add nsw i64 %indvar.middle, -1
  %tobool2.i = icmp eq i64 %indvar.middle.next, 0
  br i1 %tobool2.i, label %outermost.latch, label %middle.header

outermost.latch:                                      ; preds = %middle.latch
  %indvar.outermost.next = add nsw i32 %indvar.outermost, -5
  %tobool.i = icmp eq i32 %indvar.outermost.next, 0
  br i1 %tobool.i, label %outermost.exit, label %outermost.header

outermost.exit:                                           ; preds = %outermost.latch
  ret void
}
