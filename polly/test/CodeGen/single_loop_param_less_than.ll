; RUN: opt %loadNPMPolly '-passes=print<polly-ast>' -disable-output < %s | FileCheck %s
; RUN: opt %loadNPMPolly -passes=polly-codegen  -S < %s | FileCheck %s -check-prefix=CODEGEN
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

@A = common global [1024 x i32] zeroinitializer

define void @bar(i64 %n) {
start:
  fence seq_cst
  br label %loop.header

loop.header:
  %i = phi i64 [ 0, %start ], [ %i.next, %loop.backedge ]
  %scevgep = getelementptr [1024 x i32], ptr @A, i64 0, i64 %i
  %exitcond = icmp ne i64 %i, %n
  br i1 %exitcond, label %loop.body, label %ret

loop.body:
  store i32 1, ptr %scevgep
  br label %loop.backedge

loop.backedge:
  %i.next = add nsw i64 %i, 1
  br label %loop.header

ret:
  fence seq_cst
  ret void
}

; CHECK: for (int c0 = 0; c0 < n; c0 += 1)
; CHECK:   Stmt_loop_body(c0)

; CODEGEN: polly.start:
; CODEGEN:   br label %polly.loop_if

; CODEGEN: polly.loop_exit:
; CODEGEN:   br label %polly.merge_new_and_old

; CODEGEN: polly.loop_if:
; CODEGEN:   %polly.loop_guard = icmp slt i64 0, %n
; CODEGEN:   br i1 %polly.loop_guard, label %polly.loop_preheader, label %polly.loop_exit

; CODEGEN: polly.loop_header:
; CODEGEN:   %polly.indvar = phi i64 [ 0, %polly.loop_preheader ], [ %polly.indvar_next, %polly.stmt.loop.body ]
; CODEGEN:   br label %polly.stmt.loop.body

; CODEGEN: polly.stmt.loop.body:
; CODEGEN:   %[[offset:.*]] = shl i64 %polly.indvar, 2
; CODEGEN:   [[PTR:%[a-zA-Z0-9_\.]+]] = getelementptr i8, ptr @A, i64 %[[offset]]
; CODEGEN:   store i32 1, ptr [[PTR]]
; CODEGEN:   %polly.indvar_next = add nsw i64 %polly.indvar, 1
; CODEGEN:   %polly.loop_cond = icmp slt i64 %polly.indvar_next, %n
; CODEGEN:   br i1 %polly.loop_cond, label %polly.loop_header, label %polly.loop_exit

; CODEGEN: polly.loop_preheader:
; CODEGEN:   br label %polly.loop_header
