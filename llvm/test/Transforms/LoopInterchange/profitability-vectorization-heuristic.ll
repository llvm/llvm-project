; RUN: opt < %s -passes=loop-interchange -cache-line-size=64 \
; RUN:     -pass-remarks-output=%t -disable-output -loop-interchange-profitabilities=vectorize
; RUN: FileCheck -input-file %t %s

@A = dso_local global [256 x [256 x float]] zeroinitializer
@B = dso_local global [256 x [256 x float]] zeroinitializer
@C = dso_local global [256 x [256 x float]] zeroinitializer

; Check that the below loops are exchanged for vectorization.
;
; for (int i = 0; i < 256; i++) {
;   for (int j = 1; j < 256; j++) {
;     A[i][j] = A[i][j-1] + B[i][j];
;     C[i][j] += 1;
;   }
; }
;

; CHECK:      --- !Passed
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            Interchanged
; CHECK-NEXT: Function:        interchange_necessary_for_vectorization
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          Loop interchanged with enclosing loop.
define void @interchange_necessary_for_vectorization() {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 1, %entry ], [ %i.next, %for.i.inc ]
  br label %for.j.body

for.j.body:
  %j = phi i64 [ 1, %for.i.header ], [ %j.next, %for.j.body ]
  %j.dec = add nsw i64 %j, -1
  %a.load.index = getelementptr nuw inbounds [256 x [256 x float]], ptr @A, i64 %i, i64 %j.dec
  %b.index = getelementptr nuw inbounds [256 x [256 x float]], ptr @B, i64 %i, i64 %j
  %c.index = getelementptr nuw inbounds [256 x [256 x float]], ptr @C, i64 %i, i64 %j
  %a = load float, ptr %a.load.index, align 4
  %b = load float, ptr %b.index, align 4
  %c = load float, ptr %c.index, align 4
  %add.0 = fadd float %a, %b
  %a.store.index = getelementptr nuw inbounds [256 x [256 x float]], ptr @A, i64 %i, i64 %j
  store float %add.0, ptr %a.store.index, align 4
  %add.1 = fadd float %c, 1.0
  store float %add.1, ptr %c.index, align 4
  %j.next = add nuw nsw i64 %j, 1
  %cmp.j = icmp eq i64 %j.next, 256
  br i1 %cmp.j, label %for.i.inc, label %for.j.body

for.i.inc:
  %i.next = add nuw nsw i64 %i, 1
  %cmp.i = icmp eq i64 %i.next, 256
  br i1 %cmp.i, label %exit, label %for.i.header

exit:
  ret void
}

; Check that the following innermost loop can be vectorized so that
; interchanging is unnecessary.
;
; for (int i = 0; i < 256; i++)
;   for (int j = 1; j < 256; j++)
;     A[i][j-1] = A[i][j] + B[i][j];
;
; FIXME: These loops are exchanged at this time due to the problem in
; profitability heuristic calculation for vectorization.

; CHECK:      --- !Passed
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            Interchanged
; CHECK-NEXT: Function:        interchange_unnecesasry_for_vectorization
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          Loop interchanged with enclosing loop.
define void @interchange_unnecesasry_for_vectorization() {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 1, %entry ], [ %i.next, %for.i.inc ]
  br label %for.j.body

for.j.body:
  %j = phi i64 [ 1, %for.i.header ], [ %j.next, %for.j.body ]
  %j.dec = add nsw i64 %j, -1
  %a.load.index = getelementptr nuw inbounds [256 x [256 x float]], ptr @A, i64 %i, i64 %j
  %b.index = getelementptr nuw inbounds [256 x [256 x float]], ptr @B, i64 %i, i64 %j
  %a = load float, ptr %a.load.index, align 4
  %b = load float, ptr %b.index, align 4
  %add = fadd float %a, %b
  %a.store.index = getelementptr nuw inbounds [256 x [256 x float]], ptr @A, i64 %i, i64 %j.dec
  store float %add, ptr %a.store.index, align 4
  %j.next = add nuw nsw i64 %j, 1
  %cmp.j = icmp eq i64 %j.next, 256
  br i1 %cmp.j, label %for.i.inc, label %for.j.body

for.i.inc:
  %i.next = add nuw nsw i64 %i, 1
  %cmp.i = icmp eq i64 %i.next, 256
  br i1 %cmp.i, label %exit, label %for.i.header

exit:
  ret void
}
