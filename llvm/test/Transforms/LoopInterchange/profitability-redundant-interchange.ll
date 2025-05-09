; RUN: opt < %s -passes=loop-interchange -cache-line-size=1 -pass-remarks-output=%t -disable-output \
; RUN:      -verify-dom-info -verify-loop-info
; RUN: FileCheck -input-file %t %s


; Test that loop-interchange doesn't undo its own transoformation. This is
; the case when the cost computed by CacheCost is the same for the loop of `j`
; and `k`.
;
; #define N 4
; int a[N*N][N*N][N*N];
; void f() {
;   for (int i = 0; i < N; i++)
;     for (int j = 1; j < 2*N; j++)
;       for (int k = 1; k < 2*N; k++)
;         a[i][k+1][j-1] -= a[i+N-1][k][j];
; }

; CHECK:      --- !Passed
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            Interchanged
; CHECK-NEXT: Function:        f
; CHECK-NEXT: Args:
; CHECK-NEXT:    - String:          Loop interchanged with enclosing loop.
; CHECK-NEXT: ...
; CHECK-NEXT: --- !Missed
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            Dependence
; CHECK-NEXT: Function:        f
; CHECK-NEXT: Args:
; CHECK-NEXT:  - String:       Cannot interchange loops due to dependences.
; CHECK-NEXT: ...
; CHECK-NEXT: --- !Missed
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            InterchangeNotProfitable
; CHECK-NEXT: Function:        f
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          Interchanging loops is not considered to improve cache locality nor vectorization.
; CHECK-NEXT: ...

@a = dso_local local_unnamed_addr global [16 x [16 x [16 x i32]]] zeroinitializer, align 4

define dso_local void @f() {
entry:
  br label %for.cond1.preheader

for.cond1.preheader:
  %indvars.iv46 = phi i64 [ 0, %entry ], [ %indvars.iv.next47, %for.cond.cleanup3 ]
  %0 = add nuw nsw i64 %indvars.iv46, 3
  br label %for.cond5.preheader

for.cond5.preheader:
  %indvars.iv41 = phi i64 [ 1, %for.cond1.preheader ], [ %indvars.iv.next42, %for.cond.cleanup7 ]
  %1 = add nsw i64 %indvars.iv41, -1
  br label %for.body8

for.cond.cleanup3:
  %indvars.iv.next47 = add nuw nsw i64 %indvars.iv46, 1
  %exitcond50 = icmp ne i64 %indvars.iv.next47, 4
  br i1 %exitcond50, label %for.cond1.preheader, label %for.cond.cleanup

for.cond.cleanup7:
  %indvars.iv.next42 = add nuw nsw i64 %indvars.iv41, 1
  %exitcond45 = icmp ne i64 %indvars.iv.next42, 8
  br i1 %exitcond45, label %for.cond5.preheader, label %for.cond.cleanup3

for.body8:
  %indvars.iv = phi i64 [ 1, %for.cond5.preheader ], [ %indvars.iv.next, %for.body8 ]
  %arrayidx12 = getelementptr inbounds nuw [16 x [16 x [16 x i32]]], ptr @a, i64 0, i64 %0, i64 %indvars.iv, i64 %indvars.iv41
  %2 = load i32, ptr %arrayidx12, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %arrayidx20 = getelementptr inbounds [16 x [16 x [16 x i32]]], ptr @a, i64 0, i64 %indvars.iv46, i64 %indvars.iv.next, i64 %1
  %3 = load i32, ptr %arrayidx20, align 4
  %sub21 = sub nsw i32 %3, %2
  store i32 %sub21, ptr %arrayidx20, align 4
  %exitcond = icmp ne i64 %indvars.iv.next, 8
  br i1 %exitcond, label %for.body8, label %for.cond.cleanup7

for.cond.cleanup:
  ret void
}
