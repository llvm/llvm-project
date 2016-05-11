; RUN: opt < %s -S -loop-unroll -unroll-count=4 | FileCheck -check-prefix=CHECK_COUNT4 %s
; RUN: opt < %s -S -loop-unroll | FileCheck -check-prefix=CHECK_NOCOUNT %s


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
