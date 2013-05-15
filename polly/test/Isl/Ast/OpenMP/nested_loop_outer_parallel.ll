; RUN: opt %loadPolly -polly-ast -polly-ast-detect-parallel -analyze < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-pc-linux-gnu"

; for (i = 0; i < n; i++)
;   for (j = 0; j < n; j++)
;     A[i] = 1;

@A = common global [1024 x i32] zeroinitializer
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
  %scevgep = getelementptr [1024 x i32]* @A, i64 0, i64 %i
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

; CHECK: #pragma omp parallel for
; CHECK: for (int c1 = 0; c1 < n; c1 += 1)
; CHECK:   for (int c3 = 0; c3 < n; c3 += 1)
; CHECK:     Stmt_loop_body(c1, c3);
