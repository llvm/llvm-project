; RUN: opt %loadPolly -polly-ast -polly-ast-detect-parallel -analyze < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-pc-linux-gnu"

; for (i = 0; i < n; i++)
;   for (j = 0; j < n; j++)
;     A[i][j] = 1;

@A = common global [1024 x [1024 x i32]] zeroinitializer
define void @bar(i64 %n) {
start:
  fence seq_cst
  br label %loop.i

loop.i:
  %i = phi i64 [ 0, %start ], [ %i.next, %loop.i.backedge ]
  %exitcond.i = icmp ne i64 %i, %n
  br i1 %exitcond.i, label %loop.j, label %ret

loop.j:
  %j = phi i64 [ 0, %loop.i], [ %j.next, %loop.j.backedge ]
  %exitcond.j = icmp ne i64 %j, %n
  br i1 %exitcond.j, label %loop.body, label %loop.i.backedge

loop.body:
  %scevgep = getelementptr [1024 x [1024 x i32] ]* @A, i64 0, i64 %j, i64 %i
  store i32 1, i32* %scevgep
  br label %loop.j.backedge

loop.j.backedge:
  %j.next = add nsw i64 %j, 1
  br label %loop.j

loop.i.backedge:
  %i.next = add nsw i64 %i, 1
  br label %loop.i

ret:
  fence seq_cst
  ret void
}

; At the first look both loops seem parallel, however due to the delinearization
; we get the following dependences:
;    [n] -> { loop_body[i0, i1] -> loop_body[1024 + i0, -1 + i1]:
;                                           0 <= i0 < n - 1024  and 1 <= i1 < n}
; They cause the outer loop to be non-parallel.  We can only prove their
; absence, if we know that n < 1024. This information is currently not available
; to polly. However, we should be able to obtain it due to the out of bounds
; memory accesses, that would happen if n >= 1024.
;
; CHECK: for (int c1 = 0; c1 < n; c1 += 1)
; CHECK:   #pragma omp parallel for
; CHECK:   for (int c3 = 0; c3 < n; c3 += 1)
; CHECK:     Stmt_loop_body(c1, c3);
