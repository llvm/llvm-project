; RUN: opt < %s -passes=loop-interchange -cache-line-size=64 \
; RUN:     -pass-remarks-output=%t -disable-output
; RUN: FileCheck -input-file %t --check-prefix=PROFIT-CACHE %s

; RUN: opt < %s -passes=loop-interchange -cache-line-size=64 \
; RUN:     -pass-remarks-output=%t -disable-output -loop-interchange-profitabilities=vectorize,cache,instorder
; RUN: FileCheck -input-file %t --check-prefix=PROFIT-VEC %s

@A = dso_local global [256 x [256 x float]] zeroinitializer
@B = dso_local global [256 x [256 x float]] zeroinitializer
@C = dso_local global [256 x [256 x float]] zeroinitializer
@D = dso_local global [256 x [256 x float]] zeroinitializer
@E = dso_local global [256 x [256 x float]] zeroinitializer
@F = dso_local global [256 x [256 x float]] zeroinitializer

; Check the behavior of the LoopInterchange cost-model. In the below code,
; exchanging the loops is not profitable in terms of cache, but it is necessary
; to vectorize the innermost loop.
;
; for (int i = 0; i < 256; i++)
;   for (int j = 1; j < 256; j++)
;     A[j][i] = A[j-1][i] + B[j][i] + C[i][j] + D[i][j] + E[i][j] + F[i][j];
;

; PROFIT-CACHE:      --- !Missed
; PROFIT-CACHE-NEXT: Pass:            loop-interchange
; PROFIT-CACHE-NEXT: Name:            InterchangeNotProfitable
; PROFIT-CACHE-NEXT: Function:        f
; PROFIT-CACHE-NEXT: Args:
; PROFIT-CACHE-NEXT:   - String:          Interchanging loops is not considered to improve cache locality nor vectorization.
; PROFIT-CACHE-NEXT: ...

; PROFIT-VEC:      --- !Passed
; PROFIT-VEC-NEXT: Pass:            loop-interchange
; PROFIT-VEC-NEXT: Name:            Interchanged
; PROFIT-VEC-NEXT: Function:        f
; PROFIT-VEC-NEXT: Args:
; PROFIT-VEC-NEXT:   - String:          Loop interchanged with enclosing loop.
; PROFIT-VEC-NEXT: ...
define void @f() {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.i.inc ]
  br label %for.j.body

for.j.body:
  %j = phi i64 [ 1, %for.i.header ], [ %j.next, %for.j.body ]
  %j.dec = add nsw i64 %j, -1
  %a.0.index = getelementptr nuw inbounds [256 x [256 x float]], ptr @A, i64 0, i64 %j.dec, i64 %i
  %b.index = getelementptr nuw inbounds [256 x [256 x float]], ptr @B, i64 0, i64 %j, i64 %i
  %c.index = getelementptr nuw inbounds [256 x [256 x float]], ptr @C, i64 0, i64 %i, i64 %j
  %d.index = getelementptr nuw inbounds [256 x [256 x float]], ptr @D, i64 0, i64 %i, i64 %j
  %e.index = getelementptr nuw inbounds [256 x [256 x float]], ptr @E, i64 0, i64 %i, i64 %j
  %f.index = getelementptr nuw inbounds [256 x [256 x float]], ptr @F, i64 0, i64 %i, i64 %j
  %a.0 = load float, ptr %a.0.index, align 4
  %b = load float, ptr %b.index, align 4
  %c = load float, ptr %c.index, align 4
  %d = load float, ptr %d.index, align 4
  %e = load float, ptr %e.index, align 4
  %f = load float, ptr %f.index, align 4
  %add.0 = fadd float %a.0, %b
  %add.1 = fadd float %add.0, %c
  %add.2 = fadd float %add.1, %d
  %add.3 = fadd float %add.2, %e
  %add.4 = fadd float %add.3, %f
  %a.1.index = getelementptr nuw inbounds [256 x float], ptr @A, i64 %j, i64 %i
  store float %add.4, ptr %a.1.index, align 4
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
