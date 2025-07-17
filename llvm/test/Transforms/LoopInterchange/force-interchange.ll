; RUN: opt < %s -passes=loop-interchange -pass-remarks-output=%t -disable-output -loop-interchange-profitabilities=ignore -S
; RUN: FileCheck --input-file=%t %s

; There should be no reason to interchange this, unless it is forced.
;
;     for (int i = 0; i<1024; i++)
;       for (int j = 0; j<1024; j++)
;         A[i][j] = 42;
;
; CHECK:      --- !Passed
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            Interchanged
; CHECK-NEXT: Function:        f
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          Loop interchanged with enclosing loop.
; CHECK-NEXT: ...

@A = dso_local local_unnamed_addr global [1024 x [1024 x i32]] zeroinitializer, align 4

define dso_local void @f() local_unnamed_addr #0 {
entry:
  br label %for.cond1.preheader

for.cond1.preheader:
  %indvars.iv17 = phi i64 [ 0, %entry ], [ %indvars.iv.next18, %for.cond.cleanup3 ]
  br label %for.body4

for.cond.cleanup:
  ret void

for.cond.cleanup3:
  %indvars.iv.next18 = add nuw nsw i64 %indvars.iv17, 1
  %exitcond20.not = icmp eq i64 %indvars.iv.next18, 1024
  br i1 %exitcond20.not, label %for.cond.cleanup, label %for.cond1.preheader

for.body4:
  %indvars.iv = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next, %for.body4 ]
  %arrayidx6 = getelementptr inbounds nuw [1024 x [1024 x i32]], ptr @A, i64 0, i64 %indvars.iv17, i64 %indvars.iv
  store i32 42, ptr %arrayidx6, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup3, label %for.body4
}
