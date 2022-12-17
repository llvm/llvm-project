; RUN: opt < %s -S -passes=loop-unroll -unroll-count=4 | FileCheck -check-prefix=CHECK_COUNT4 %s
; RUN: opt < %s -S -passes=loop-unroll | FileCheck -check-prefix=CHECK_NOCOUNT %s
; RUN: opt < %s -S -passes='require<profile-summary>,function(loop-unroll)' -pgso | FileCheck -check-prefix=PGSO %s
; RUN: opt < %s -S -passes='require<profile-summary>,function(loop-unroll)' -pgso=false | FileCheck -check-prefix=NPGSO %s


;///////////////////// TEST 1 //////////////////////////////

; This test shows that the loop is unrolled according to the specified
; unroll factor.

define void @Test1() nounwind {
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %inc, %loop ]
  %inc = add i32 %iv, 1
  %exitcnd = icmp uge i32 %inc, 1024
  br i1 %exitcnd, label %exit, label %loop

exit:
  ret void
}

; CHECK_COUNT4-LABEL: @Test1
; CHECK_COUNT4:      phi
; CHECK_COUNT4-NEXT: add
; CHECK_COUNT4-NEXT: add
; CHECK_COUNT4-NEXT: add
; CHECK_COUNT4-NEXT: add
; CHECK_COUNT4-NEXT: icmp


;///////////////////// TEST 2 //////////////////////////////

; This test shows that with optnone attribute, the loop is not unrolled
; even if an unroll factor was specified.

define void @Test2() nounwind optnone noinline {
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %inc, %loop ]
  %inc = add i32 %iv, 1
  %exitcnd = icmp uge i32 %inc, 1024
  br i1 %exitcnd, label %exit, label %loop

exit:
  ret void
}

; CHECK_COUNT4-LABEL: @Test2
; CHECK_COUNT4:      phi
; CHECK_COUNT4-NEXT: add
; CHECK_COUNT4-NEXT: icmp


;///////////////////// TEST 3 //////////////////////////////

; This test shows that this loop is fully unrolled by default.

@tab = common global [24 x i32] zeroinitializer, align 4

define i32 @Test3() {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.05 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds [24 x i32], [24 x i32]* @tab, i32 0, i32 %i.05
  store i32 %i.05, i32* %arrayidx, align 4
  %inc = add nuw nsw i32 %i.05, 1
  %exitcond = icmp eq i32 %inc, 24
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret i32 42
}

; CHECK_NOCOUNT-LABEL: @Test3
; CHECK_NOCOUNT:      store
; CHECK_NOCOUNT-NEXT: store
; CHECK_NOCOUNT-NEXT: store
; CHECK_NOCOUNT-NEXT: store
; CHECK_NOCOUNT-NEXT: store
; CHECK_NOCOUNT-NEXT: store
; CHECK_NOCOUNT-NEXT: store
; CHECK_NOCOUNT-NEXT: store
; CHECK_NOCOUNT-NEXT: store
; CHECK_NOCOUNT-NEXT: store
; CHECK_NOCOUNT-NEXT: store
; CHECK_NOCOUNT-NEXT: store
; CHECK_NOCOUNT-NEXT: store
; CHECK_NOCOUNT-NEXT: store
; CHECK_NOCOUNT-NEXT: store
; CHECK_NOCOUNT-NEXT: store
; CHECK_NOCOUNT-NEXT: store
; CHECK_NOCOUNT-NEXT: store
; CHECK_NOCOUNT-NEXT: store
; CHECK_NOCOUNT-NEXT: store
; CHECK_NOCOUNT-NEXT: store
; CHECK_NOCOUNT-NEXT: store
; CHECK_NOCOUNT-NEXT: store
; CHECK_NOCOUNT-NEXT: store
; CHECK_NOCOUNT-NEXT: ret


;///////////////////// TEST 4 //////////////////////////////

; This test shows that with optsize attribute, this loop is not unrolled.

define i32 @Test4() optsize {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.05 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds [24 x i32], [24 x i32]* @tab, i32 0, i32 %i.05
  store i32 %i.05, i32* %arrayidx, align 4
  %inc = add nuw nsw i32 %i.05, 1
  %exitcond = icmp eq i32 %inc, 24
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret i32 42
}

; CHECK_NOCOUNT-LABEL: @Test4
; CHECK_NOCOUNT:      phi
; CHECK_NOCOUNT:      icmp

;///////////////////// TEST 5 //////////////////////////////

; This test shows that with PGO, this loop is cold and not unrolled.

define i32 @Test5() !prof !14 {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.05 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds [24 x i32], [24 x i32]* @tab, i32 0, i32 %i.05
  store i32 %i.05, i32* %arrayidx, align 4
  %inc = add nuw nsw i32 %i.05, 1
  %exitcond = icmp eq i32 %inc, 24
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret i32 42
}

; PGSO-LABEL: @Test5
; PGSO:      phi
; PGSO:      icmp
; NPGSO-LABEL: @Test5
; NPGSO-NOT:      phi
; NPGSO-NOT:      icmp

;///////////////////// TEST 6 //////////////////////////////

; This test tests that unroll hints take precedence over PGSO and that this loop
; gets unrolled even though it's cold.

define i32 @Test6() !prof !14 {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.05 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds [24 x i32], [24 x i32]* @tab, i32 0, i32 %i.05
  store i32 %i.05, i32* %arrayidx, align 4
  %inc = add nuw nsw i32 %i.05, 1
  %exitcond = icmp eq i32 %inc, 24
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !15

for.end:                                          ; preds = %for.body
  ret i32 42
}

; PGSO-LABEL: @Test6
; PGSO:      store
; PGSO:      store
; PGSO:      store
; PGSO:      store
; NPGSO-LABEL: @Test6
; NPGSO:      store
; NPGSO:      store
; NPGSO:      store
; NPGSO:      store

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
!15 = !{!15, !16}
!16 = !{!"llvm.loop.unroll.count", i32 4}
