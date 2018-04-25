; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s
;
; The SCoP contains a loop with multiple exit blocks (BBs after leaving
; the loop). The current implementation of deriving their domain derives
; only a common domain for all of the exit blocks. We disabled loops with
; multiple exit blocks until this is fixed.
; XFAIL: *
;
; CHECK:      Statements {
; CHECK-NEXT:     Stmt_for_body_outer
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             { Stmt_for_body_outer[i0] : 0 <= i0 <= 257 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             { Stmt_for_body_outer[i0] -> [i0, 0, 0, 0] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             { Stmt_for_body_outer[i0] -> MemRef_A[i0] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             { Stmt_for_body_outer[i0] -> MemRef_A[i0] };
; CHECK-NEXT:     Stmt_for_body
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             { Stmt_for_body[257, i1] : 0 <= i1 <= 1025; Stmt_for_body[i0, 0] : 0 <= i0 <= 256 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             { Stmt_for_body[257, i1] -> [257, 1, i1, 0]; Stmt_for_body[i0, 0] -> [i0, 1, 0, 0] : i0 <= 256 };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             { Stmt_for_body[i0, i1] -> MemRef_A[i1] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             { Stmt_for_body[i0, i1] -> MemRef_A[i1] };
; CHECK-NEXT:     Stmt_for_inc
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             { Stmt_for_inc[257, i1] : 0 <= i1 <= 1025 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             { Stmt_for_inc[i0, i1] -> [257, 1, i1, 1] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             { Stmt_for_inc[i0, i1] -> MemRef_A[i1] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             { Stmt_for_inc[i0, i1] -> MemRef_A[i1] };
; CHECK-NEXT: }

target datalayout = "e-m:e-i64:64-i128:128-n8:16:32:64-S128"

define void @foo(i32* %A) {
entry:
  br label %for.body.outer

for.body.outer:                                   ; preds = %for.body, %entry
  %indvar = phi i32 [0, %entry], [%indvar.next, %for.body]
  %addr = getelementptr i32, i32* %A, i32 %indvar
  %val = load i32, i32* %addr
  %indvar.next = add i32 %indvar, 1
  store i32 %val, i32* %addr
  br label %for.body

for.body:                                         ; preds = %for.inc, %for.body.outer
  %indvar.2 = phi i32 [0, %for.body.outer], [%indvar.2.next, %for.inc]
  %addr.2 = getelementptr i32, i32* %A, i32 %indvar.2
  %val.2  = load i32, i32* %addr.2
  %indvar.2.next = add i32 %indvar.2, 1
  store i32 %val.2, i32* %addr.2
  %cond.1 = icmp sle i32 %indvar, 256
  br i1 %cond.1, label %for.body.outer, label %for.inc

for.inc:                                          ; preds = %for.body
  %addr.3 = getelementptr i32, i32* %A, i32 %indvar.2
  %val.3  = load i32, i32* %addr.3
  store i32 %val.3, i32* %addr.3
  %cond = icmp sle i32 %indvar.2, 1024
  br i1 %cond, label %for.body, label %for.end

for.end:                                          ; preds = %for.inc
  ret void
}
