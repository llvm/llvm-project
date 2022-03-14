; RUN: opt %loadPolly -polly-print-ast -polly-parallel -polly-dependences-computeout=1 -disable-output < %s | FileCheck %s
; RUN: opt %loadPolly -print-polyhedral-info -polly-check-parallel -disable-output < %s | FileCheck %s -check-prefix=PINFO
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

; for (i = 0; i < n; i++)
;   A[i] = 1;

@A = common global [1024 x i32] zeroinitializer
define void @bar(i64 %n) {
start:
  fence seq_cst
  br label %loop.header

loop.header:
  %i = phi i64 [ 0, %start ], [ %i.next, %loop.backedge ]
  %scevgep = getelementptr [1024 x i32], [1024 x i32]* @A, i64 0, i64 %i
  %exitcond = icmp ne i64 %i, %n
  br i1 %exitcond, label %loop.body, label %ret

loop.body:
  store i32 1, i32* %scevgep
  br label %loop.backedge

loop.backedge:
  %i.next = add nsw i64 %i, 1
  br label %loop.header

ret:
  fence seq_cst
  ret void
}

; CHECK-NOT: #pragma simd
; CHECK-NOT: #pragma omp parallel for
; CHECK: for (int c0 = 0; c0 < n; c0 += 1)
; CHECK:   Stmt_loop_body(c0)
; PINFO: loop.header: Loop is parallel.
