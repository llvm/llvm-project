; RUN: opt -opaque-pointers=0 %loadPolly -polly-print-ast -disable-output < %s | FileCheck %s
; RUN: opt -opaque-pointers=0 %loadPolly -polly-codegen  -S < %s | FileCheck %s -check-prefix=CODEGEN
; RUN: opt -opaque-pointers=0 %loadPolly -polly-codegen < %s | opt -opaque-pointers=0 -passes='print<loops>' -disable-output 2>&1 | FileCheck %s -check-prefix=LOOPS
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

@A = common global [1024 x i32] zeroinitializer

define void @bar(i64 %n) {
start:
  %n_plus_one = add i64 %n, 1
  fence seq_cst
  br label %loop.header

loop.header:
  %i = phi i64 [ 0, %start ], [ %i.next, %loop.backedge ]
  %scevgep = getelementptr [1024 x i32], [1024 x i32]* @A, i64 0, i64 %i
  %exitcond = icmp ne i64 %i, %n_plus_one
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

; CHECK: for (int c0 = 0; c0 <= n; c0 += 1)
; CHECK:   Stmt_loop_body(c0)

; CODEGEN: polly.start:
; CODEGEN:   br label %polly.loop_if

; CODEGEN: polly.loop_exit:
; CODEGEN:   br label %polly.merge_new_and_old

; CODEGEN: polly.loop_if:
; CODEGEN:   %polly.loop_guard = icmp sle i64 0, %n
; CODEGEN:   br i1 %polly.loop_guard, label %polly.loop_preheader, label %polly.loop_exit

; CODEGEN: polly.loop_header:
; CODEGEN:   %polly.indvar = phi i64 [ 0, %polly.loop_preheader ], [ %polly.indvar_next, %polly.stmt.loop.body ]
; CODEGEN:   br label %polly.stmt.loop.body

; CODEGEN: polly.stmt.loop.body:
; CODEGEN:   [[PTR:%[a-zA-Z0-9_\.]+]] = getelementptr [1024 x i32], [1024 x i32]* @A, i64 0, i64 %polly.indvar
; CODEGEN:   store i32 1, i32* [[PTR]]
; CODEGEN:   %polly.indvar_next = add nsw i64 %polly.indvar, 1
; CODEGEN:   %polly.loop_cond = icmp sle i64 %polly.indvar_next, %n
; CODEGEN:   br i1 %polly.loop_cond, label %polly.loop_header, label %polly.loop_exit

; CODEGEN: polly.loop_preheader:
; CODEGEN:   br label %polly.loop_header

; LOOPS-DAG: Loop at depth 1 containing: %loop.header<header><exiting>,%loop.body,%loop.backedge<latch>
; LOOPS-DAG: Loop at depth 1 containing: %polly.loop_header<header>,%polly.stmt.loop.body<latch><exiting>
