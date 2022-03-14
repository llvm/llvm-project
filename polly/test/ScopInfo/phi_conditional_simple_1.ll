; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s
;
;    void jd(int *A, int c) {
;      for (int i = 0; i < 1024; i++) {
;        if (c)
;          A[i] = 1;
;        else
;          A[i] = 2;
;      }
;    }
;
; CHECK:    Statements {
; CHECK-LABEL:      Stmt_if_else
; CHECK-NOT: Access
; CHECK:            MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK:                [c] -> { Stmt_if_else[i0] -> MemRef_phi__phi[] };
; CHECK-NOT: Access
; CHECK-LABEL:      Stmt_if_then
; CHECK-NOT: Access
; CHECK:            MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK:                [c] -> { Stmt_if_then[i0] -> MemRef_phi__phi[] };
; CHECK-NOT: Access
; CHECK-LABEL:      Stmt_if_end
; CHECK-NOT: Access
; CHECK:            ReadAccess := [Reduction Type: NONE] [Scalar: 1]
; CHECK:                [c] -> { Stmt_if_end[i0] -> MemRef_phi__phi[] };
; CHECK-NOT: Access
; CHECK:            MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK:                [c] -> { Stmt_if_end[i0] -> MemRef_A[i0] };
; CHECK-NOT: Access
; CHECK:    }

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @jd(i32* %A, i32 %c) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i64 %indvars.iv, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %tobool = icmp eq i32 %c, 0
  br i1 %tobool, label %if.else, label %if.then

if.then:                                          ; preds = %for.body
  br label %if.end

if.else:                                          ; preds = %for.body
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %phi = phi i32 [ 1, %if.then], [ 2, %if.else ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  store i32 %phi, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
