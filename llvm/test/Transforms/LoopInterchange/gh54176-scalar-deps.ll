; Remove 'S' Scalar Dependencies #119345
; Scalar dependencies are not handled correctly, so they were removed to avoid
; miscompiles. The loop nest in this test case used to be interchanged, but it's
; no longer triggering. XFAIL'ing this test to indicate that this test should
; interchanged if scalar deps are handled correctly.
;
; XFAIL: *

; RUN: opt < %s -passes=loop-interchange -pass-remarks-output=%t -disable-output
; RUN: FileCheck -input-file %t %s

@f = dso_local local_unnamed_addr global [4 x [9 x i32]] [[9 x i32] [i32 5, i32 3, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0], [9 x i32] zeroinitializer, [9 x i32] zeroinitializer, [9 x i32] zeroinitializer], align 4
@g = common dso_local local_unnamed_addr global i32 0, align 4

; int32_t f[3][3];
; int32_t g;
;
; int32_t test1(_Bool cond) {
;     for (int64_t i = 0; i < 3; i++) {
;         for (int64_t j = 0; j < 3; j++) {
;            int32_t val = f[i][j];
;            if (val == 0)
;              if (!cond)
;                g++;
;            else
;              g = 3;
;              if (!cond)
;                g++;
;         }
;     }
;     return g;
; }
;
define dso_local i32 @test1(i1 %cond) {
;
; FIXME: if there's an output dependency inside the loop and Src doesn't
; dominate Dst, we should not interchange. Thus, this currently miscompiles.
;
; CHECK:        --- !Passed
; CHECK-NEXT:   Pass:            loop-interchange
; CHECK-NEXT:   Name:            Interchanged
; CHECK-NEXT:   Function:        test1
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - String:          Loop interchanged with enclosing loop.
; CHECK-NEXT: ...
;
for.preheader:
  br label %outerloop

outerloop:
  %i = phi i64 [ 0, %for.preheader ], [ %indvars.iv.next21.i, %for.latch ]
  br label %innerloop

innerloop:
  %j = phi i64 [ 0, %outerloop ], [ %j.next, %if.end ]
  %arrayidx6.i = getelementptr inbounds [4 x [9 x i32]], [4 x [9 x i32]]* @f, i64 0, i64 %j, i64 %i
  %i1 = load i32, i32* %arrayidx6.i, align 4
  %tobool.i = icmp eq i32 %i1, 0
  br i1 %tobool.i, label %land.end, label %land.rhs

land.rhs:
  store i32 3, i32* @g, align 4
  br label %land.end

land.end:
  br i1 %cond, label %if.end, label %if.then

if.then:
  %i2 = load i32, i32* @g, align 4
  %inc.i = add i32 %i2, 1
  store i32 %inc.i, i32* @g, align 4
  br label %if.end

if.end:
  %j.next = add nuw nsw i64 %j, 1
  %exitcond.i = icmp eq i64 %j.next, 3
  br i1 %exitcond.i, label %for.latch, label %innerloop

for.latch:
  %indvars.iv.next21.i = add nsw i64 %i, 1
  %cmp.i = icmp slt i64 %i, 2
  br i1 %cmp.i, label %outerloop, label %exit

exit:
  %i3 = load i32, i32* @g, align 4
  ret i32 %i3
}

; int32_t f[3][3];
; int32_t g;
;
; int32_t test2(_Bool cond) {
;     for (int64_t i = 0; i < 3; i++) {
;       for (int64_t j = 0; j < 3; j++) {
;          int32_t val = f[i][j];
;          g = 3;
;          if (val == 0)
;            if (!cond)
;              g++;
;          else
;            if (!cond)
;              g++;
;         }
;     }
;     return g;
; }
;
define dso_local i32 @test2(i1 %cond) {
;
; FIXME: if there's an output dependency inside the loop and Src doesn't
; dominate Dst, we should not interchange. Thus, this currently miscompiles.
;
; CHECK:        --- !Passed
; CHECK-NEXT:   Pass:            loop-interchange
; CHECK-NEXT:   Name:            Interchanged
; CHECK-NEXT:   Function:        test2
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - String:          Loop interchanged with enclosing loop.
; CHECK-NEXT: ...
;
for.preheader:
  br label %outerloop

outerloop:
  %i = phi i64 [ 0, %for.preheader ], [ %indvars.iv.next21.i, %for.latch ]
  br label %innerloop

innerloop:
  %j = phi i64 [ 0, %outerloop ], [ %j.next, %if.end ]
  %arrayidx6.i = getelementptr inbounds [4 x [9 x i32]], [4 x [9 x i32]]* @f, i64 0, i64 %j, i64 %i
  %i1 = load i32, i32* %arrayidx6.i, align 4
  %tobool.i = icmp eq i32 %i1, 0
  store i32 3, i32* @g, align 4
  br i1 %tobool.i, label %land.end, label %land.rhs

land.rhs:
  br label %land.end

land.end:
  br i1 %cond, label %if.end, label %if.then

if.then:
  %i2 = load i32, i32* @g, align 4
  %inc.i = add i32 %i2, 1
  store i32 %inc.i, i32* @g, align 4
  br label %if.end

if.end:
  %j.next = add nuw nsw i64 %j, 1
  %exitcond.i = icmp eq i64 %j.next, 3
  br i1 %exitcond.i, label %for.latch, label %innerloop

for.latch:
  %indvars.iv.next21.i = add nsw i64 %i, 1
  %cmp.i = icmp slt i64 %i, 2
  br i1 %cmp.i, label %outerloop, label %exit

exit:
  %i3 = load i32, i32* @g, align 4
  ret i32 %i3
}
