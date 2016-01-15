; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s
;
;    int jd(int *restrict A, int x, int N, int c) {
;      for (int i = 0; i < N; i++)
;        for (int j = 0; j < N; j++)
;          if (i < c)
;            x += A[i];
;      return x;
;    }

; CHECK:      Statements {
; CHECK-NEXT:     Stmt_for_cond
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [N, c] -> { Stmt_for_cond[i0] : 0 <= i0 <= N; Stmt_for_cond[0] : N < 0 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [N, c] -> { Stmt_for_cond[i0] -> [i0, 0, 0, 0] : i0 <= N; Stmt_for_cond[0] -> [0, 0, 0, 0] : N < 0 };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [N, c] -> { Stmt_for_cond[i0] -> MemRef_x_addr_0__phi[] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [N, c] -> { Stmt_for_cond[i0] -> MemRef_x_addr_0[] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [N, c] -> { Stmt_for_cond[i0] -> MemRef_A[i0] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [N, c] -> { Stmt_for_cond[i0] -> MemRef_x_addr_0[] };
; CHECK-NEXT:     Stmt_for_body
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [N, c] -> { Stmt_for_body[i0] : 0 <= i0 < N };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [N, c] -> { Stmt_for_body[i0] -> [i0, 1, 0, 0] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [N, c] -> { Stmt_for_body[i0] -> MemRef_x_addr_0[] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [N, c] -> { Stmt_for_body[i0] -> MemRef_x_addr_1__phi[] };
; CHECK-NEXT:     Stmt_for_cond1
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [N, c] -> { Stmt_for_cond1[i0, i1] : 0 <= i0 < N and 0 <= i1 <= N };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [N, c] -> { Stmt_for_cond1[i0, i1] -> [i0, 2, i1, 0] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [N, c] -> { Stmt_for_cond1[i0, i1] -> MemRef_x_addr_1[] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [N, c] -> { Stmt_for_cond1[i0, i1] -> MemRef_x_addr_1__phi[] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [N, c] -> { Stmt_for_cond1[i0, i1] -> MemRef_x_addr_1[] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [N, c] -> { Stmt_for_cond1[i0, i1] -> MemRef_x_addr_1[] };
; CHECK-NEXT:     Stmt_for_body3
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [N, c] -> { Stmt_for_body3[i0, i1] : 0 <= i0 < N and 0 <= i1 < N };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [N, c] -> { Stmt_for_body3[i0, i1] -> [i0, 2, i1, 1] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [N, c] -> { Stmt_for_body3[i0, i1] -> MemRef_x_addr_1[] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [N, c] -> { Stmt_for_body3[i0, i1] -> MemRef_x_addr_2__phi[] };
; CHECK-NEXT:     Stmt_if_then
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [N, c] -> { Stmt_if_then[i0, i1] : 0 <= i0 < c and i0 < N and 0 <= i1 < N };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [N, c] -> { Stmt_if_then[i0, i1] -> [i0, 2, i1, 2] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [N, c] -> { Stmt_if_then[i0, i1] -> MemRef_x_addr_1[] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [N, c] -> { Stmt_if_then[i0, i1] -> MemRef_A[i0] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [N, c] -> { Stmt_if_then[i0, i1] -> MemRef_x_addr_2__phi[] };
; CHECK-NEXT:     Stmt_if_end
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [N, c] -> { Stmt_if_end[i0, i1] : 0 <= i0 < N and 0 <= i1 < N };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [N, c] -> { Stmt_if_end[i0, i1] -> [i0, 2, i1, 3] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [N, c] -> { Stmt_if_end[i0, i1] -> MemRef_x_addr_2[] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [N, c] -> { Stmt_if_end[i0, i1] -> MemRef_x_addr_2__phi[] };
; CHECK-NEXT:     Stmt_for_inc
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [N, c] -> { Stmt_for_inc[i0, i1] : 0 <= i0 < N and 0 <= i1 < N };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [N, c] -> { Stmt_for_inc[i0, i1] -> [i0, 2, i1, 4] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [N, c] -> { Stmt_for_inc[i0, i1] -> MemRef_x_addr_2[] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [N, c] -> { Stmt_for_inc[i0, i1] -> MemRef_x_addr_1__phi[] };
; CHECK-NEXT:     Stmt_for_inc5
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [N, c] -> { Stmt_for_inc5[i0] : 0 <= i0 < N };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [N, c] -> { Stmt_for_inc5[i0] -> [i0, 3, 0, 0] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [N, c] -> { Stmt_for_inc5[i0] -> MemRef_x_addr_1[] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [N, c] -> { Stmt_for_inc5[i0] -> MemRef_x_addr_0__phi[] };
; CHECK-NEXT: }

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define i32 @jd(i32* noalias %A, i32 %x, i32 %N, i32 %c) {
entry:
  %tmp = sext i32 %N to i64
  %tmp1 = sext i32 %c to i64
  br label %for.cond

for.cond:                                         ; preds = %for.inc5, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc5 ], [ 0, %entry ]
  %x.addr.0 = phi i32 [ %x, %entry ], [ %x.addr.1, %for.inc5 ]
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  store i32 %x.addr.0, i32* %arrayidx2
  %cmp = icmp slt i64 %indvars.iv, %tmp
  br i1 %cmp, label %for.body, label %for.end7

for.body:                                         ; preds = %for.cond
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %x.addr.1 = phi i32 [ %x.addr.0, %for.body ], [ %x.addr.2, %for.inc ]
  %j.0 = phi i32 [ 0, %for.body ], [ %inc, %for.inc ]
  %exitcond = icmp ne i32 %j.0, %N
  br i1 %exitcond, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  %cmp4 = icmp slt i64 %indvars.iv, %tmp1
  br i1 %cmp4, label %if.then, label %if.end

if.then:                                          ; preds = %for.body3
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp2 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %x.addr.1, %tmp2
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body3
  %x.addr.2 = phi i32 [ %add, %if.then ], [ %x.addr.1, %for.body3 ]
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %inc = add nsw i32 %j.0, 1
  br label %for.cond1

for.end:                                          ; preds = %for.cond1
  br label %for.inc5

for.inc5:                                         ; preds = %for.end
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end7:                                         ; preds = %for.cond
  ret i32 %x.addr.0
}

