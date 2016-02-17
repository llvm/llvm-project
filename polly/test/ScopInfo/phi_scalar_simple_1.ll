; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s
;
; The assumed context should be empty since the <nsw> flags on the IV
; increments already guarantee that there is no wrap in the loop trip
; count.
;
;    int jd(int *restrict A, int x, int N) {
;      for (int i = 1; i < N; i++)
;        for (int j = 3; j < N; j++)
;          x += A[i];
;      return x;
;    }

; CHECK:      Assumed Context:
; CHECK-NEXT: [N] -> {  :  }
;
; CHECK:      Statements {
; CHECK-NEXT:     Stmt_for_cond
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [N] -> { Stmt_for_cond[i0] : N >= 2 and 0 <= i0 < N; Stmt_for_cond[0] : N <= 1 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [N] -> { Stmt_for_cond[i0] -> [i0, 0, 0, 0] : N >= 2 and i0 < N; Stmt_for_cond[0] -> [0, 0, 0, 0] : N <= 1 };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [N] -> { Stmt_for_cond[i0] -> MemRef_x_addr_0__phi[] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [N] -> { Stmt_for_cond[i0] -> MemRef_x_addr_0[] };
; CHECK-NEXT:     Stmt_for_body
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [N] -> { Stmt_for_body[i0] : N >= 2 and 0 <= i0 <= -2 + N };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [N] -> { Stmt_for_body[i0] -> [i0, 1, 0, 0] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [N] -> { Stmt_for_body[i0] -> MemRef_x_addr_0[] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [N] -> { Stmt_for_body[i0] -> MemRef_x_addr_1__phi[] };
; CHECK-NEXT:     Stmt_for_cond1
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [N] -> { Stmt_for_cond1[i0, i1] : 0 <= i0 <= -2 + N and 0 <= i1 <= -3 + N };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [N] -> { Stmt_for_cond1[i0, i1] -> [i0, 2, i1, 0] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [N] -> { Stmt_for_cond1[i0, i1] -> MemRef_x_addr_1__phi[] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [N] -> { Stmt_for_cond1[i0, i1] -> MemRef_x_addr_1[] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [N] -> { Stmt_for_cond1[i0, i1] -> MemRef_x_addr_1_lcssa__phi[] };
; CHECK-NEXT:     Stmt_for_inc
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [N] -> { Stmt_for_inc[i0, i1] : 0 <= i0 <= -2 + N and 0 <= i1 <= -4 + N };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [N] -> { Stmt_for_inc[i0, i1] -> [i0, 2, i1, 1] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [N] -> { Stmt_for_inc[i0, i1] -> MemRef_x_addr_1__phi[] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [N] -> { Stmt_for_inc[i0, i1] -> MemRef_A[1 + i0] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [N] -> { Stmt_for_inc[i0, i1] -> MemRef_x_addr_1[] };
; CHECK-NEXT:     Stmt_for_end
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [N] -> { Stmt_for_end[i0] : N >= 3 and 0 <= i0 <= -2 + N };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [N] -> { Stmt_for_end[i0] -> [i0, 3, 0, 0] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [N] -> { Stmt_for_end[i0] -> MemRef_x_addr_1_lcssa[] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [N] -> { Stmt_for_end[i0] -> MemRef_x_addr_1_lcssa__phi[] };
; CHECK-NEXT:     Stmt_for_inc4
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [N] -> { Stmt_for_inc4[i0] : N >= 3 and 0 <= i0 <= -2 + N };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [N] -> { Stmt_for_inc4[i0] -> [i0, 4, 0, 0] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [N] -> { Stmt_for_inc4[i0] -> MemRef_x_addr_1_lcssa[] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [N] -> { Stmt_for_inc4[i0] -> MemRef_x_addr_0__phi[] };
; CHECK-NEXT: }

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define i32 @jd(i32* noalias %A, i32 %x, i32 %N) {
entry:
  %tmp = sext i32 %N to i64
  br label %for.cond

for.cond:                                         ; preds = %for.inc4, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc4 ], [ 1, %entry ]
  %x.addr.0 = phi i32 [ %x, %entry ], [ %x.addr.1.lcssa, %for.inc4 ]
  %cmp = icmp slt i64 %indvars.iv, %tmp
  br i1 %cmp, label %for.body, label %for.end6

for.body:                                         ; preds = %for.cond
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %x.addr.1 = phi i32 [ %x.addr.0, %for.body ], [ %add, %for.inc ]
  %j.0 = phi i32 [ 3, %for.body ], [ %inc, %for.inc ]
  %exitcond = icmp ne i32 %j.0, %N
  br i1 %exitcond, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  br label %for.inc

for.inc:                                          ; preds = %for.body3
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp1 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %x.addr.1, %tmp1
  %inc = add nsw i32 %j.0, 1
  br label %for.cond1

for.end:                                          ; preds = %for.cond1
  %x.addr.1.lcssa = phi i32 [ %x.addr.1, %for.cond1 ]
  br label %for.inc4

for.inc4:                                         ; preds = %for.end
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end6:                                         ; preds = %for.cond
  ret i32 %x.addr.0
}
