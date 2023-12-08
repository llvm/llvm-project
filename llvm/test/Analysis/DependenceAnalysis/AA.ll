; RUN: opt < %s -disable-output "-passes=print<da>"                            \
; RUN: "-aa-pipeline=basic-aa,tbaa" 2>&1 | FileCheck %s

; CHECK-LABEL: 'Dependence Analysis' for function 'test_no_noalias'
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
define void @test_no_noalias(ptr %A, ptr %B) {
  store i32 1, ptr %A
  store i32 2, ptr %B
  ret void
}

; CHECK-LABEL: test_one_noalias
; CHECK: da analyze - none!
; CHECK: da analyze - none!
; CHECK: da analyze - none!
define void @test_one_noalias(ptr noalias %A, ptr %B) {
  store i32 1, ptr %A
  store i32 2, ptr %B
  ret void
}

; CHECK-LABEL: test_two_noalias
; CHECK: da analyze - none!
; CHECK: da analyze - none!
; CHECK: da analyze - none!
define void @test_two_noalias(ptr noalias %A, ptr noalias %B) {
  store i32 1, ptr %A
  store i32 2, ptr %B
  ret void
}

; CHECK-LABEL: test_global_alias
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
@g = global i32 5
define void @test_global_alias(ptr %A) {
  store i32 1, ptr %A
  store i32 2, ptr @g
  ret void
}

; CHECK-LABEL: test_global_noalias
; CHECK: da analyze - none!
; CHECK: da analyze - none!
; CHECK: da analyze - none!
define void @test_global_noalias(ptr noalias %A) {
  store i32 1, ptr %A
  store i32 2, ptr @g
  ret void
}

; CHECK-LABEL: test_global_size
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

@a = global i16 5, align 2
@b = global ptr @a, align 4
define void @test_global_size() {
  %l0 = load ptr, ptr @b, align 4
  %l1 = load i16, ptr %l0, align 2
  store i16 1, ptr @a, align 2
  ret void
}

; CHECK-LABEL: test_tbaa_same
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
define void @test_tbaa_same(ptr %A, ptr %B) {
  store i32 1, ptr %A, !tbaa !5
  store i32 2, ptr %B, !tbaa !5
  ret void
}

; CHECK-LABEL: test_tbaa_diff
; CHECK: da analyze - none!
; CHECK: da analyze - none!
; CHECK: da analyze - none!
define void @test_tbaa_diff(ptr %A, ptr %B) {
  store i32 1, ptr %A, !tbaa !5
  store i16 2, ptr %B, !tbaa !9
  ret void
}

; CHECK-LABEL: tbaa_loop
; CHECK: da analyze - input
; CHECK: da analyze - none
; CHECK: da analyze - output
define void @tbaa_loop(i32 %I, i32 %J, ptr nocapture %A, ptr nocapture readonly %B) {
entry:
  %cmp = icmp ne i32 %J, 0
  %cmp122 = icmp ne i32 %I, 0
  %or.cond = and i1 %cmp, %cmp122
  br i1 %or.cond, label %for.outer.preheader, label %for.end

for.outer.preheader:
  br label %for.outer

for.outer:
  %i.us = phi i32 [ %add8.us, %for.latch ], [ 0, %for.outer.preheader ]
  br label %for.inner

for.inner:
  %j.us = phi i32 [ 0, %for.outer ], [ %inc.us, %for.inner ]
  %sum1.us = phi i32 [ 0, %for.outer ], [ %add.us, %for.inner ]
  %arrayidx.us = getelementptr inbounds i16, ptr %B, i32 %j.us
  %0 = load i16, ptr %arrayidx.us, align 4, !tbaa !9
  %sext = sext i16 %0 to i32
  %add.us = add i32 %sext, %sum1.us
  %inc.us = add nuw i32 %j.us, 1
  %exitcond = icmp eq i32 %inc.us, %J
  br i1 %exitcond, label %for.latch, label %for.inner

for.latch:
  %add.us.lcssa = phi i32 [ %add.us, %for.inner ]
  %arrayidx6.us = getelementptr inbounds i32, ptr %A, i32 %i.us
  store i32 %add.us.lcssa, ptr %arrayidx6.us, align 4, !tbaa !5
  %add8.us = add nuw i32 %i.us, 1
  %exitcond25 = icmp eq i32 %add8.us, %I
  br i1 %exitcond25, label %for.end.loopexit, label %for.outer

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}

!5 = !{!6, !6, i64 0}
!6 = !{!"int", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
!9 = !{!10, !10, i64 0}
!10 = !{!"short", !7, i64 0}
