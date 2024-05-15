; RUN: opt %loadPolly -S -polly-codegen < %s | FileCheck %s
;
;    int jd(int *restrict A, int x, int N, int c) {
;      for (int i = 0; i < N; i++)
;        for (int j = 0; j < N; j++)
;          if (i < c)
;            x += A[i];
;      return x;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define i32 @jd(ptr noalias %A, i32 %x, i32 %N, i32 %c) {
entry:
; CHECK-LABEL: entry:
; CHECK-DAG:     %x.addr.2.s2a = alloca i32
; CHECK-DAG:     %x.addr.2.phiops = alloca i32
; CHECK-DAG:     %x.addr.1.s2a = alloca i32
; CHECK-DAG:     %x.addr.1.phiops = alloca i32
; CHECK-DAG:     %x.addr.0.s2a = alloca i32
; CHECK-DAG:     %x.addr.0.phiops = alloca i32
  %tmp = sext i32 %N to i64
  %tmp1 = sext i32 %c to i64
  br label %for.cond

; CHECK-LABEL: polly.merge_new_and_old:
; CHECK:         %x.addr.0.merge = phi i32 [ %x.addr.0.final_reload, %polly.exiting ], [ %x.addr.0, %for.cond ]
; CHECK:         ret i32 %x.addr.0.merge

; CHECK-LABEL: polly.start:
; CHECK-NEXT:    store i32 %x, ptr %x.addr.0.phiops
; CHECK-NEXT:    sext

; CHECK-LABEL: polly.merge{{[a-z_0-9]*}}:
; CHECK:         %x.addr.0.final_reload = load i32, ptr %x.addr.0.s2a

for.cond:                                         ; preds = %for.inc5, %entry
; CHECK-LABEL: polly.stmt.for.cond{{[0-9]*}}:
; CHECK:         %x.addr.0.phiops.reload[[R1:[0-9]*]] = load i32, ptr %x.addr.0.phiops
; CHECK:         store i32 %x.addr.0.phiops.reload[[R1]], ptr %x.addr.0.s2a
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc5 ], [ 0, %entry ]
  %x.addr.0 = phi i32 [ %x, %entry ], [ %x.addr.1, %for.inc5 ]
  %cmp = icmp slt i64 %indvars.iv, %tmp
  br i1 %cmp, label %for.body, label %for.end7

for.body:                                         ; preds = %for.cond
; CHECK-LABEL: polly.stmt.for.body:
; CHECK:         %x.addr.0.s2a.reload[[R2:[0-9]*]] = load i32, ptr %x.addr.0.s2a
; CHECK:         store i32 %x.addr.0.s2a.reload[[R2]], ptr %x.addr.1.phiops
  br label %for.cond1

for.inc5:                                         ; preds = %for.end
; CHECK-LABEL: polly.stmt.for.inc5:
; CHECK:         %x.addr.1.s2a.reload[[R5:[0-9]*]] = load i32, ptr %x.addr.1.s2a
; CHECK:         store i32 %x.addr.1.s2a.reload[[R5]], ptr %x.addr.0.phiops
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.cond1:                                        ; preds = %for.inc, %for.body
; CHECK-LABEL: polly.stmt.for.cond1:
; CHECK:         %x.addr.1.phiops.reload = load i32, ptr %x.addr.1.phiops
; CHECK:         store i32 %x.addr.1.phiops.reload, ptr %x.addr.1.s2a
  %x.addr.1 = phi i32 [ %x.addr.0, %for.body ], [ %x.addr.2, %for.inc ]
  %j.0 = phi i32 [ 0, %for.body ], [ %inc, %for.inc ]
  %exitcond = icmp ne i32 %j.0, %N
  br i1 %exitcond, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
; CHECK-LABEL: polly.stmt.for.body3:
; CHECK:  %x.addr.1.s2a.reload = load i32, ptr %x.addr.1.s2a
; CHECK:  store i32 %x.addr.1.s2a.reload, ptr %x.addr.2.phiops
  %cmp4 = icmp slt i64 %indvars.iv, %tmp1
  br i1 %cmp4, label %if.then, label %if.end

if.end:                                           ; preds = %if.then, %for.body3
; CHECK-LABEL: polly.stmt.if.end:
; CHECK:         %x.addr.2.phiops.reload = load i32, ptr %x.addr.2.phiops
; CHECK:         store i32 %x.addr.2.phiops.reload, ptr %x.addr.2.s2a
  %x.addr.2 = phi i32 [ %add, %if.then ], [ %x.addr.1, %for.body3 ]
  br label %for.inc

for.inc:                                          ; preds = %if.end
; CHECK-LABEL: polly.stmt.for.inc:
; CHECK:         %x.addr.2.s2a.reload[[R3:[0-9]*]] = load i32, ptr %x.addr.2.s2a
; CHECK:         store i32 %x.addr.2.s2a.reload[[R3]], ptr %x.addr.1.phiops
  %inc = add nsw i32 %j.0, 1
  br label %for.cond1

if.then:                                          ; preds = %for.body3
; CHECK-LABEL: polly.stmt.if.then:
; CHECK:         %x.addr.1.s2a.reload[[R5:[0-9]*]] = load i32, ptr %x.addr.1.s2a
; CHECK:         %p_add = add nsw i32 %x.addr.1.s2a.reload[[R5]], %tmp2_p_scalar_
; CHECK:         store i32 %p_add, ptr %x.addr.2.phiops
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %tmp2 = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %x.addr.1, %tmp2
  br label %if.end

for.end:                                          ; preds = %for.cond1
  br label %for.inc5

for.end7:                                         ; preds = %for.cond
  ret i32 %x.addr.0
}

