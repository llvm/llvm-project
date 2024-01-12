; REQUIRES: asserts
; RUN: opt < %s -passes=loop-interchange -cache-line-size=64 -verify-dom-info -verify-loop-info \
; RUN:     -S -debug 2>&1 | FileCheck %s

;; Loops should not be interchanged in this case as the store is wider than
;; array element type
;;
;;  char p[7];
;;  for (int j = 0; j < 2; ++j)
;;    for (int i = 0; i < 2; ++i)
;;      *((int*)&p[2*i+j]) = 2*i+j+1;

; CHECK: Loads or Stores Type i32 is wider than the element type i8
; CHECK: Populating dependency matrix failed

define i32 @main() {
entry:
  %p = alloca [7 x i8], align 1
  br label %for.cond1.preheader

; Loop:
for.cond1.preheader:                              ; preds = %entry, %for.cond.cleanup3
  %indvars.iv29 = phi i64 [ 0, %entry ], [ %indvars.iv.next30, %for.cond.cleanup3 ]
  br label %for.body4

for.cond.cleanup3:                                ; preds = %for.body4
  %indvars.iv.next30 = add nuw nsw i64 %indvars.iv29, 1
  %cmp = icmp eq i64 %indvars.iv29, 0
  br i1 %cmp, label %for.cond1.preheader, label %for.cond.cleanup

for.body4:                                        ; preds = %for.cond1.preheader, %for.body4
  %indvars.iv = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next, %for.body4 ]
  %0 = shl nuw nsw i64 %indvars.iv, 1
  %1 = add nuw nsw i64 %0, %indvars.iv29
  %2 = add nuw nsw i64 %1, 1
  %arrayidx = getelementptr inbounds [7 x i8], ptr %p, i64 0, i64 %1
  %3 = trunc i64 %2 to i32
  store i32 %3, ptr %arrayidx, align 1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %cmp2 = icmp eq i64 %indvars.iv, 0
  br i1 %cmp2, label %for.body4, label %for.cond.cleanup3

; Exit blocks
for.cond.cleanup:                                 ; preds = %for.cond.cleanup3
  ret i32 0
}
