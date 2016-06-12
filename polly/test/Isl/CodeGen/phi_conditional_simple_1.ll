; RUN: opt %loadPolly -analyze -polly-ast < %s | FileCheck %s --check-prefix=AST
; RUN: opt %loadPolly -S -polly-codegen < %s | FileCheck %s
;
;    void jd(int *A, int c) {
;      for (int i = 0; i < 1024; i++) {
;        if (c)
;          A[i] = 1;
;        else
;          A[i] = 2;
;      }
;    }

; AST:    for (int c0 = 0; c0 <= 1023; c0 += 1) {
; AST:      if (c <= -1 || c >= 1) {
; AST:        Stmt_if_then(c0);
; AST:      } else
; AST:        Stmt_if_else(c0);
; AST:      Stmt_if_end(c0);
; AST:    }
;
; CHECK-LABEL:  entry:
; CHECK-NEXT:     %phi.phiops = alloca i32
; CHECK-LABEL:  polly.stmt.if.end:
; CHECK-NEXT:     %phi.phiops.reload = load i32, i32* %phi.phiops
; CHECK-NEXT:     %scevgep
; CHECK-NEXT:     store i32 %phi.phiops.reload, i32*
; CHECK-LABEL:  polly.stmt.if.then:
; CHECK-NEXT:     store i32 1, i32* %phi.phiops
; CHECK-NEXT:     br label %polly.merge{{[.]?}}
; CHECK-LABEL:  polly.stmt.if.else:
; CHECK-NEXT:     store i32 2, i32* %phi.phiops
; CHECK-NEXT:     br label %polly.merge{{[.]?}}
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @jd(i32* %A, i32 %c) {
entry:
  br label %for.cond

for.cond:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i64 %indvars.iv, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.body:
  %tobool = icmp eq i32 %c, 0
  br i1 %tobool, label %if.else, label %if.then

if.then:
  br label %if.end

if.else:
  br label %if.end

if.end:
  %phi = phi i32 [ 1, %if.then], [ 2, %if.else ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  store i32 %phi, i32* %arrayidx, align 4
  br label %for.inc

for.inc:
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:
  ret void
}
