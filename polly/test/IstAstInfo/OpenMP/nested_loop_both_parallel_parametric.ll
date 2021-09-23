; RUN: opt %loadPolly -polly-ast -polly-parallel -polly-parallel-force -analyze < %s | FileCheck %s
; RUN: opt %loadPolly -polyhedral-info -polly-check-parallel -analyze < %s | FileCheck %s -check-prefix=PINFO
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
; int A[1024][1024];
; void bar(int n) {
;   for (i = 0; i < n; i++)
;     for (j = 0; j < n; j++)
;       A[i][j] = 1;
; }
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
  %scevgep = getelementptr [1024 x [1024 x i32] ], [1024 x [1024 x i32] ]* @A, i64 0, i64 %j, i64 %i
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

; CHECK: if (n <= 1024)
; CHECK:   #pragma omp parallel for
; CHECK:   for (int c0 = 0; c0 < n; c0 += 1)
; CHECK:     #pragma simd
; CHECK:     for (int c1 = 0; c1 < n; c1 += 1)
; CHECK:       Stmt_loop_body(c0, c1);

; PINFO:      loop.i: Loop is parallel.
; PINFO-NEXT: loop.j: Loop is parallel.
