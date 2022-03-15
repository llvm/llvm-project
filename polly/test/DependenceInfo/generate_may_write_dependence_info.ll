; RUN: opt -S %loadPolly -polly-print-dependences -disable-output < %s | FileCheck %s -check-prefix=VALUE
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

;                     for (int i = 0; i < N; i++) {
; A.must.write.20:      A[i] = 20;
;
; compute.i.square:    if (i * i)
; A.may.write.90:         A[i] = 90;
;
; B.write.from.A:       B[i] = A[i];
; A.must.write.42:      A[i] = 42;
;                     }
define void @f(i32* %A, i32* %B) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i64 %indvars.iv, 3000
  br i1 %exitcond, label %A.must.write.20, label %for.end

A.must.write.20:
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  store i32 20, i32* %arrayidx, align 4
  br label %compute.i.square

compute.i.square:
  %tmp = mul nsw i64 %indvars.iv, %indvars.iv
  %tmp2 = trunc i64 %tmp to i32
  %tobool = icmp eq i32 %tmp2, 0
  br i1 %tobool, label %B.write.from.A, label %A.may.write.90

A.may.write.90:
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  store i32 90, i32* %arrayidx2, align 4
  br label %B.write.from.A

B.write.from.A:
  %arrayidx4 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp3 = load i32, i32* %arrayidx4, align 4
  %arrayidx6 = getelementptr inbounds i32, i32* %B, i64 %indvars.iv
  store i32 %tmp3, i32* %arrayidx6, align 4
  br label %A.must.write.42
  ; br label %for.inc

A.must.write.42:
  %arrayidx5 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  store i32 42, i32* %arrayidx5, align 4
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
; VALUE: RAW dependences:
; VALUE-NEXT:   { Stmt_compute_i_square__TO__B_write_from_A[i0] -> Stmt_B_write_from_A[i0] : 0 <= i0 <= 2999; Stmt_A_must_write_20[i0] -> Stmt_B_write_from_A[i0] : 0 <= i0 <= 2999 }
; VALUE-NEXT: WAR dependences:
; VALUE-NEXT:   { Stmt_B_write_from_A[i0] -> Stmt_A_must_write_42[i0] : 0 <= i0 <= 2999 }
; VALUE-NEXT: WAW dependences:
; VALUE-NEXT:   { Stmt_A_must_write_20[i0] -> Stmt_compute_i_square__TO__B_write_from_A[i0] : 0 <= i0 <= 2999; Stmt_compute_i_square__TO__B_write_from_A[i0] -> Stmt_A_must_write_42[i0] : 0 <= i0 <= 2999; Stmt_A_must_write_20[i0] -> Stmt_A_must_write_42[i0] : 0 <= i0 <= 2999 }
; VALUE-NEXT: Reduction dependences:
; VALUE-NEXT:   {  }
; VALUE-NEXT: Transitive closure of reduction dependences:
; VALUE-NEXT:   {  }
