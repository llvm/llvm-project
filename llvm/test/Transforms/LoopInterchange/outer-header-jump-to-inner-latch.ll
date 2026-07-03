; Remove 'S' Scalar Dependencies #119345
; Scalar dependencies are not handled correctly, so they were removed to avoid
; miscompiles. The loop nest in this test case used to be interchanged, but it's
; no longer triggering. XFAIL'ing this test to indicate that this test should
; interchanged if scalar deps are handled correctly.
;
; XFAIL: *

; RUN: opt -passes=loop-interchange -cache-line-size=64 -verify-dom-info -verify-loop-info -verify-loop-lcssa %s -pass-remarks-output=%t -disable-output
; RUN: FileCheck -input-file %t %s

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
;
; CHECK:       --- !Passed
; CHECK-NEXT:  Pass:            loop-interchange
; CHECK-NEXT:  Name:            Interchanged
; CHECK-NEXT:  Function:        test1
; CHECK-NEXT:  Args:
; CHECK-NEXT:    - String:          Loop interchanged with enclosing loop.
; CHECK-NEXT:  ...
;
define void @test1() {
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
;
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
;
; CHECK:       --- !Passed
; CHECK-NEXT:  Pass:            loop-interchange
; CHECK-NEXT:  Name:            Interchanged
; CHECK-NEXT:  Function:        test2
; CHECK-NEXT:  Args:
; CHECK-NEXT:    - String:          Loop interchanged with enclosing loop.
; CHECK-NEXT:  ...
;
define void @test2() {
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
