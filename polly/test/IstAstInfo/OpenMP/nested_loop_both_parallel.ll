; RUN: opt %loadPolly -polly-print-ast -polly-parallel -disable-output < %s | FileCheck %s
; RUN: opt %loadPolly -print-polyhedral-info -polly-check-parallel -disable-output < %s | FileCheck %s -check-prefix=PINFO
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

; for (i = 0; i < 1024; i++)
;   for (j = 0; j < 1024; j++)
;     A[i][j] = 1;

@A = common global [1024 x [1024 x i32]] zeroinitializer
define void @bar() {
start:
  fence seq_cst
  br label %loop.i

loop.i:
  %i = phi i64 [ 0, %start ], [ %i.next, %loop.i.backedge ]
  %exitcond.i = icmp ne i64 %i, 1024
  br i1 %exitcond.i, label %loop.j, label %ret

loop.j:
  %j = phi i64 [ 0, %loop.i], [ %j.next, %loop.j.backedge ]
  %exitcond.j = icmp ne i64 %j, 1024
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

; Make sure we do not accidentally generate nested openmp parallel for
; annotations.

; CHECK:     #pragma omp parallel for
; CHECK:     for (int c0 = 0; c0 <= 1023; c0 += 1)
; CHECK-NOT:   #pragma omp parallel for
; CHECK:       #pragma simd
; CHECK-NOT:   #pragma omp parallel for
; CHECK:       for (int c1 = 0; c1 <= 1023; c1 += 1)
; CHECK:         Stmt_loop_body(c0, c1);
;
; PINFO:      loop.i: Loop is parallel.
; PINFO-NEXT: loop.j: Loop is parallel.
