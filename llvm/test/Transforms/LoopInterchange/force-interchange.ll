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
  br label %outer.header

outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %inner.header ]
  br label %inner.body

inner.header:
  %i.next = add nuw nsw i64 %i, 1
  %exitcond20.not = icmp eq i64 %i.next, 1024
  br i1 %exitcond20.not, label %exit, label %outer.header

inner.body:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.body ]
  %arrayidx6 = getelementptr inbounds nuw [1024 x [1024 x i32]], ptr @A, i64 0, i64 %i, i64 %j
  store i32 42, ptr %arrayidx6, align 4
  %j.next = add nuw nsw i64 %j, 1
  %exitcond.not = icmp eq i64 %j.next, 1024
  br i1 %exitcond.not, label %inner.header, label %inner.body

exit:
  ret void
}
