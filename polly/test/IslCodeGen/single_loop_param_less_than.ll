; RUN: opt %loadPolly -polly-ast -analyze < %s | FileCheck %s
; RUN: opt %loadPolly -polly-codegen-isl -S < %s | FileCheck %s -check-prefix=CODEGEN
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-pc-linux-gnu"

@A = common global [1024 x i32] zeroinitializer

define void @bar(i64 %n) {
start:
  fence seq_cst
  br label %loop.header

loop.header:
  %i = phi i64 [ 0, %start ], [ %i.next, %loop.backedge ]
  %scevgep = getelementptr [1024 x i32]* @A, i64 0, i64 %i
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

; CHECK: for (int c1 = 0; c1 < n; c1 += 1)
; CHECK:   Stmt_loop_body(c1)

; CODEGEN: polly.start:
; CODEGEN:   br label %polly.loop_header

; CODEGEN: polly.loop_after:
; CODEGEN:   br label %polly.merge_new_and_old

; CODEGEN: polly.loop_header:
; CODEGEN:   %polly.loopiv = phi i64 [ 0, %polly.start ], [ %polly.next_loopiv, %polly.stmt.loop.body ]
; CODEGEN:   %polly.next_loopiv = add nsw i64 %polly.loopiv, 1
; CODEGEN:   %0 = icmp slt i64 %polly.loopiv, %n
; CODEGEN:   br i1 %0, label %polly.loop_body, label %polly.loop_after

; CODEGEN: polly.loop_body:
; CODEGEN:   br label %polly.stmt.loop.body

; CODEGEN: polly.stmt.loop.body:
; CODEGEN:   %p_scevgep.moved.to.loop.body = getelementptr [1024 x i32]* @A, i64 0, i64 %polly.loopiv
; CODEGEN:   store i32 1, i32* %p_scevgep.moved.to.loop.body
; CODEGEN:   br label %polly.loop_header
