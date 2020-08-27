; This test verifies that the loop vectorizer will NOT produce a tail
; loop with the optimize for size or the minimize size attributes.
; REQUIRES: asserts
; RUN: opt < %s -loop-vectorize -S | FileCheck %s
; RUN: opt < %s -loop-vectorize -pgso -S | FileCheck %s -check-prefix=PGSO
; RUN: opt < %s -loop-vectorize -pgso=false -S | FileCheck %s -check-prefix=NPGSO

target datalayout = "E-m:e-p:32:32-i64:32-f64:32:64-a:0:32-n32-S128"

@tab = common global [32 x i8] zeroinitializer, align 1

define i32 @foo_optsize() #0 {
; CHECK-LABEL: @foo_optsize(
; CHECK-NOT: <2 x i8>
; CHECK-NOT: <4 x i8>

entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.08 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds [32 x i8], [32 x i8]* @tab, i32 0, i32 %i.08
  %0 = load i8, i8* %arrayidx, align 1
  %cmp1 = icmp eq i8 %0, 0
  %. = select i1 %cmp1, i8 2, i8 1
  store i8 %., i8* %arrayidx, align 1
  %inc = add nsw i32 %i.08, 1
  %exitcond = icmp eq i32 %i.08, 202
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret i32 0
}

attributes #0 = { optsize }

define i32 @foo_minsize() #1 {
; CHECK-LABEL: @foo_minsize(
; CHECK-NOT: <2 x i8>
; CHECK-NOT: <4 x i8>
; CHECK-LABEL: @foo_pgso(

entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.08 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds [32 x i8], [32 x i8]* @tab, i32 0, i32 %i.08
  %0 = load i8, i8* %arrayidx, align 1
  %cmp1 = icmp eq i8 %0, 0
  %. = select i1 %cmp1, i8 2, i8 1
  store i8 %., i8* %arrayidx, align 1
  %inc = add nsw i32 %i.08, 1
  %exitcond = icmp eq i32 %i.08, 202
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret i32 0
}

attributes #1 = { minsize }

define i32 @foo_pgso() !prof !14 {
; PGSO-LABEL: @foo_pgso(
; PGSO-NOT: <{{[0-9]+}} x i8>
; NPGSO-LABEL: @foo_pgso(
; NPGSO: <{{[0-9]+}} x i8>

entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.08 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds [32 x i8], [32 x i8]* @tab, i32 0, i32 %i.08
  %0 = load i8, i8* %arrayidx, align 1
  %cmp1 = icmp eq i8 %0, 0
  %. = select i1 %cmp1, i8 2, i8 1
  store i8 %., i8* %arrayidx, align 1
  %inc = add nsw i32 %i.08, 1
  %exitcond = icmp eq i32 %i.08, 202
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret i32 0
}

; PR43371: don't run into an assert due to emitting SCEV runtime checks
; with OptForSize.
;
@cm_array = external global [2592 x i16], align 1

define void @pr43371() optsize {
;
; CHECK-LABEL: @pr43371
; CHECK-NOT:   vector.scevcheck
;
; We do not want to generate SCEV predicates when optimising for size, because
; that will lead to extra code generation such as the SCEV overflow runtime
; checks. Not generating SCEV predicates can still result in vectorisation as
; the non-consecutive loads/stores can be scalarized:
;
; CHECK: vector.body:
; CHECK: store i16 0, i16* %{{.*}}, align 1
; CHECK: store i16 0, i16* %{{.*}}, align 1
; CHECK: br i1 {{.*}}, label %vector.body
;
entry:
  br label %for.body29

for.cond.cleanup28:
  unreachable

for.body29:
  %i24.0170 = phi i16 [ 0, %entry], [ %inc37, %for.body29]
  %add33 = add i16 undef, %i24.0170
  %idxprom34 = zext i16 %add33 to i32
  %arrayidx35 = getelementptr [2592 x i16], [2592 x i16] * @cm_array, i32 0, i32 %idxprom34
  store i16 0, i16 * %arrayidx35, align 1
  %inc37 = add i16 %i24.0170, 1
  %cmp26 = icmp ult i16 %inc37, 756
  br i1 %cmp26, label %for.body29, label %for.cond.cleanup28
}

define void @pr43371_pgso() !prof !14 {
;
; CHECK-LABEL: @pr43371_pgso
; CHECK-NOT:   vector.scevcheck
;
; We do not want to generate SCEV predicates when optimising for size, because
; that will lead to extra code generation such as the SCEV overflow runtime
; checks. Not generating SCEV predicates can still result in vectorisation as
; the non-consecutive loads/stores can be scalarized:
;
; CHECK: vector.body:
; CHECK: store i16 0, i16* %{{.*}}, align 1
; CHECK: store i16 0, i16* %{{.*}}, align 1
; CHECK: br i1 {{.*}}, label %vector.body
;
entry:
  br label %for.body29

for.cond.cleanup28:
  unreachable

for.body29:
  %i24.0170 = phi i16 [ 0, %entry], [ %inc37, %for.body29]
  %add33 = add i16 undef, %i24.0170
  %idxprom34 = zext i16 %add33 to i32
  %arrayidx35 = getelementptr [2592 x i16], [2592 x i16] * @cm_array, i32 0, i32 %idxprom34
  store i16 0, i16 * %arrayidx35, align 1
  %inc37 = add i16 %i24.0170, 1
  %cmp26 = icmp ult i16 %inc37, 756
  br i1 %cmp26, label %for.body29, label %for.cond.cleanup28
}

; PR45526: don't vectorize with fold-tail if first-order-recurrence is live-out.
;
define i32 @pr45526() optsize {
;
; CHECK-LABEL: @pr45526
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %loop
; CHECK-EMPTY:
; CHECK-NEXT: loop:
; CHECK-NEXT:   %piv = phi i32 [ 0, %entry ], [ %pivPlus1, %loop ]
; CHECK-NEXT:   %for = phi i32 [ 5, %entry ], [ %pivPlus1, %loop ]
; CHECK-NEXT:   %pivPlus1 = add nuw nsw i32 %piv, 1
; CHECK-NEXT:   %cond = icmp ult i32 %piv, 510
; CHECK-NEXT:   br i1 %cond, label %loop, label %exit
; CHECK-EMPTY:
; CHECK-NEXT: exit:
; CHECK-NEXT:   %for.lcssa = phi i32 [ %for, %loop ]
; CHECK-NEXT:   ret i32 %for.lcssa
;
entry:
  br label %loop

loop:
  %piv = phi i32 [ 0, %entry ], [ %pivPlus1, %loop ]
  %for = phi i32 [ 5, %entry ], [ %pivPlus1, %loop ]
  %pivPlus1 = add nuw nsw i32 %piv, 1
  %cond = icmp ult i32 %piv, 510
  br i1 %cond, label %loop, label %exit

exit:
  ret i32 %for
}

define i32 @pr45526_pgso() !prof !14 {
;
; CHECK-LABEL: @pr45526_pgso
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %loop
; CHECK-EMPTY:
; CHECK-NEXT: loop:
; CHECK-NEXT:   %piv = phi i32 [ 0, %entry ], [ %pivPlus1, %loop ]
; CHECK-NEXT:   %for = phi i32 [ 5, %entry ], [ %pivPlus1, %loop ]
; CHECK-NEXT:   %pivPlus1 = add nuw nsw i32 %piv, 1
; CHECK-NEXT:   %cond = icmp ult i32 %piv, 510
; CHECK-NEXT:   br i1 %cond, label %loop, label %exit
; CHECK-EMPTY:
; CHECK-NEXT: exit:
; CHECK-NEXT:   %for.lcssa = phi i32 [ %for, %loop ]
; CHECK-NEXT:   ret i32 %for.lcssa
;
entry:
  br label %loop

loop:
  %piv = phi i32 [ 0, %entry ], [ %pivPlus1, %loop ]
  %for = phi i32 [ 5, %entry ], [ %pivPlus1, %loop ]
  %pivPlus1 = add nuw nsw i32 %piv, 1
  %cond = icmp ult i32 %piv, 510
  br i1 %cond, label %loop, label %exit

exit:
  ret i32 %for
}

; PR46228: Vectorize w/o versioning for unit stride under optsize and enabled
; vectorization.

; NOTE: Some assertions have been autogenerated by utils/update_test_checks.py
define void @stride1(i16* noalias %B, i32 %BStride) optsize {
; CHECK-LABEL: @stride1(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 false, label [[SCALAR_PH:%.*]], label [[VECTOR_PH:%.*]]
; CHECK:       vector.ph:
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <2 x i32> undef, i32 [[BSTRIDE:%.*]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <2 x i32> [[BROADCAST_SPLATINSERT]], <2 x i32> undef, <2 x i32> zeroinitializer
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i32 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], [[PRED_STORE_CONTINUE2:%.*]] ]
; CHECK-NEXT:    [[VEC_IND:%.*]] = phi <2 x i32> [ <i32 0, i32 1>, [[VECTOR_PH]] ], [ [[VEC_IND_NEXT:%.*]], [[PRED_STORE_CONTINUE2]] ]
; CHECK-NEXT:    [[TMP0:%.*]] = mul nsw <2 x i32> [[VEC_IND]], [[BROADCAST_SPLAT]]
; CHECK-NEXT:    [[TMP1:%.*]] = icmp ule <2 x i32> [[VEC_IND]], <i32 1024, i32 1024>
; CHECK-NEXT:    [[TMP2:%.*]] = extractelement <2 x i1> [[TMP1]], i32 0
; CHECK-NEXT:    br i1 [[TMP2]], label [[PRED_STORE_IF:%.*]], label [[PRED_STORE_CONTINUE:%.*]]
; CHECK:       pred.store.if:
; CHECK-NEXT:    [[TMP3:%.*]] = extractelement <2 x i32> [[TMP0]], i32 0
; CHECK-NEXT:    [[TMP4:%.*]] = getelementptr inbounds i16, i16* [[B:%.*]], i32 [[TMP3]]
; CHECK-NEXT:    store i16 42, i16* [[TMP4]], align 4
; CHECK-NEXT:    br label [[PRED_STORE_CONTINUE]]
; CHECK:       pred.store.continue:
; CHECK-NEXT:    [[TMP5:%.*]] = extractelement <2 x i1> [[TMP1]], i32 1
; CHECK-NEXT:    br i1 [[TMP5]], label [[PRED_STORE_IF1:%.*]], label [[PRED_STORE_CONTINUE2]]
; CHECK:       pred.store.if1:
; CHECK-NEXT:    [[TMP6:%.*]] = extractelement <2 x i32> [[TMP0]], i32 1
; CHECK-NEXT:    [[TMP7:%.*]] = getelementptr inbounds i16, i16* [[B]], i32 [[TMP6]]
; CHECK-NEXT:    store i16 42, i16* [[TMP7]], align 4
; CHECK-NEXT:    br label [[PRED_STORE_CONTINUE2]]
; CHECK:       pred.store.continue2:
; CHECK-NEXT:    [[INDEX_NEXT]] = add i32 [[INDEX]], 2
; CHECK-NEXT:    [[VEC_IND_NEXT]] = add <2 x i32> [[VEC_IND]], <i32 2, i32 2>
; CHECK-NEXT:    [[TMP8:%.*]] = icmp eq i32 [[INDEX_NEXT]], 1026
; CHECK-NEXT:    br i1 [[TMP8]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop !21
; CHECK:       middle.block:
; CHECK-NEXT:    br i1 true, label [[FOR_END:%.*]], label [[SCALAR_PH]]
; CHECK:       scalar.ph:
; CHECK:       for.end:
; CHECK-NEXT:    ret void
;
; PGSO-LABEL: @stride1(
; PGSO-NEXT:  entry:
; PGSO-NEXT:    br i1 false, label %scalar.ph, label %vector.ph
;
; NPGSO-LABEL: @stride1(
; NPGSO-NEXT:  entry:
; NPGSO-NEXT:    br i1 false, label %scalar.ph, label %vector.ph

entry:
  br label %for.body

for.body:
  %iv = phi i32 [ %iv.next, %for.body ], [ 0, %entry ]
  %mulB = mul nsw i32 %iv, %BStride
  %gepOfB = getelementptr inbounds i16, i16* %B, i32 %mulB
  store i16 42, i16* %gepOfB, align 4
  %iv.next = add nuw nsw i32 %iv, 1
  %exitcond = icmp eq i32 %iv.next, 1025
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !15

for.end:
  ret void
}

; Vectorize with versioning for unit stride for PGSO and enabled vectorization.
;
define void @stride1_pgso(i16* noalias %B, i32 %BStride) !prof !14 {
; CHECK-LABEL: @stride1_pgso(
; CHECK: vector.body
;
; PGSO-LABEL: @stride1_pgso(
; PGSO: vector.body
;
; NPGSO-LABEL: @stride1_pgso(
; NPGSO: vector.body

entry:
  br label %for.body

for.body:
  %iv = phi i32 [ %iv.next, %for.body ], [ 0, %entry ]
  %mulB = mul nsw i32 %iv, %BStride
  %gepOfB = getelementptr inbounds i16, i16* %B, i32 %mulB
  store i16 42, i16* %gepOfB, align 4
  %iv.next = add nuw nsw i32 %iv, 1
  %exitcond = icmp eq i32 %iv.next, 1025
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !15

for.end:
  ret void
}

; PR46652: Check that the need for stride==1 check prevents vectorizing a loop
; having tiny trip count, when compiling w/o -Os/-Oz.
; CHECK-LABEL: @pr46652
; CHECK-NOT: vector.scevcheck
; CHECK-NOT: vector.body
; CHECK-LABEL: for.body

@g = external global [1 x i16], align 1

define void @pr46652(i16 %stride) {
entry:
  br label %for.body

for.body:                                        ; preds = %for.body, %entry
  %l1.02 = phi i16 [ 1, %entry ], [ %inc9, %for.body ]
  %mul = mul nsw i16 %l1.02, %stride
  %arrayidx6 = getelementptr inbounds [1 x i16], [1 x i16]* @g, i16 0, i16 %mul
  %0 = load i16, i16* %arrayidx6, align 1
  %inc9 = add nuw nsw i16 %l1.02, 1
  %exitcond.not = icmp eq i16 %inc9, 16
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:                                        ; preds = %for.body
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"ProfileSummary", !1}
!1 = !{!2, !3, !4, !5, !6, !7, !8, !9}
!2 = !{!"ProfileFormat", !"InstrProf"}
!3 = !{!"TotalCount", i64 10000}
!4 = !{!"MaxCount", i64 10}
!5 = !{!"MaxInternalCount", i64 1}
!6 = !{!"MaxFunctionCount", i64 1000}
!7 = !{!"NumCounts", i64 3}
!8 = !{!"NumFunctions", i64 3}
!9 = !{!"DetailedSummary", !10}
!10 = !{!11, !12, !13}
!11 = !{i32 10000, i64 100, i32 1}
!12 = !{i32 999000, i64 100, i32 1}
!13 = !{i32 999999, i64 1, i32 2}
!14 = !{!"function_entry_count", i64 0}
!15 = distinct !{!15, !16}
!16 = !{!"llvm.loop.vectorize.enable", i1 true}
