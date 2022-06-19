; RUN: opt < %s -disable-output "-passes=print<da>"                            \
; RUN: "-aa-pipeline=basic-aa,tbaa" 2>&1 | FileCheck %s

; CHECK-LABEL: 'Dependence Analysis' for function 'test_no_noalias'
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
define void @test_no_noalias(i32* %A, i32* %B) {
  store i32 1, i32* %A
  store i32 2, i32* %B
  ret void
}

; CHECK-LABEL: test_one_noalias
; CHECK: da analyze - none!
; CHECK: da analyze - none!
; CHECK: da analyze - none!
define void @test_one_noalias(i32* noalias %A, i32* %B) {
  store i32 1, i32* %A
  store i32 2, i32* %B
  ret void
}

; CHECK-LABEL: test_two_noalias
; CHECK: da analyze - none!
; CHECK: da analyze - none!
; CHECK: da analyze - none!
define void @test_two_noalias(i32* noalias %A, i32* noalias %B) {
  store i32 1, i32* %A
  store i32 2, i32* %B
  ret void
}

; CHECK-LABEL: test_global_alias
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
@g = global i32 5
define void @test_global_alias(i32* %A) {
  store i32 1, i32* %A
  store i32 2, i32* @g
  ret void
}

; CHECK-LABEL: test_global_noalias
; CHECK: da analyze - none!
; CHECK: da analyze - none!
; CHECK: da analyze - none!
define void @test_global_noalias(i32* noalias %A) {
  store i32 1, i32* %A
  store i32 2, i32* @g
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
@b = global i16* @a, align 4
define void @test_global_size() {
  %l0 = load i16*, i16** @b, align 4
  %l1 = load i16, i16* %l0, align 2
  store i16 1, i16* @a, align 2
  ret void
}

; CHECK-LABEL: test_tbaa_same
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
define void @test_tbaa_same(i32* %A, i32* %B) {
  store i32 1, i32* %A, !tbaa !5
  store i32 2, i32* %B, !tbaa !5
  ret void
}

; CHECK-LABEL: test_tbaa_diff
; CHECK: da analyze - none!
; CHECK: da analyze - none!
; CHECK: da analyze - none!
define void @test_tbaa_diff(i32* %A, i16* %B) {
  store i32 1, i32* %A, !tbaa !5
  store i16 2, i16* %B, !tbaa !9
  ret void
}

; CHECK-LABEL: tbaa_loop
; CHECK: da analyze - input
; CHECK: da analyze - none
; CHECK: da analyze - output
define void @tbaa_loop(i32 %I, i32 %J, i32* nocapture %A, i16* nocapture readonly %B) {
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
  %arrayidx.us = getelementptr inbounds i16, i16* %B, i32 %j.us
  %0 = load i16, i16* %arrayidx.us, align 4, !tbaa !9
  %sext = sext i16 %0 to i32
  %add.us = add i32 %sext, %sum1.us
  %inc.us = add nuw i32 %j.us, 1
  %exitcond = icmp eq i32 %inc.us, %J
  br i1 %exitcond, label %for.latch, label %for.inner

for.latch:
  %add.us.lcssa = phi i32 [ %add.us, %for.inner ]
  %arrayidx6.us = getelementptr inbounds i32, i32* %A, i32 %i.us
  store i32 %add.us.lcssa, i32* %arrayidx6.us, align 4, !tbaa !5
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
