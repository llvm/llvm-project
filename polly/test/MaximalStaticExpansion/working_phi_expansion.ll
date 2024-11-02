; RUN: opt %loadPolly -polly-mse -polly-print-scops -disable-output < %s | FileCheck %s
; RUN: opt %loadNPMPolly "-passes=scop(print<polly-mse>)" -disable-output < %s | FileCheck %s
; RUN: opt %loadPolly -polly-mse -polly-print-scops -pass-remarks-analysis="polly-mse" -disable-output < %s 2>&1 | FileCheck %s --check-prefix=MSE
; RUN: opt %loadNPMPolly "-passes=scop(print<polly-mse>)" -pass-remarks-analysis="polly-mse" -disable-output < %s 2>&1 | FileCheck %s --check-prefix=MSE
;
; Verify that the accesses are correctly expanded for MemoryKind::PHI
; tmp_04 is not expanded because it need copy-in.
;
; Original source code :
;
; #define Ni 10000
; #define Nj 10000
;
; void mse(double A[Ni], double B[Nj]) {
;   int i,j;
;   double tmp = 6;
;   for (i = 0; i < Ni; i++) {
;     for (int j = 0; j<Nj; j++) {
;       tmp = tmp + 2;
;     }
;     B[i] = tmp;
;   }
;
; Check that the pass detects that tmp_04 reads from original value.
;
; MSE: MemRef_tmp_04__phi read from its original value.
;
; Check that the SAI are created except the expanded SAI of tmp_04.
;
; CHECK-NOT: double MemRef_tmp_04__phi_Stmt_for_body_expanded[10000]; // Element size 8
; CHECK: double MemRef_tmp_11__phi_Stmt_for_inc_expanded[10000][10000]; // Element size
; CHECK: double MemRef_add_lcssa__phi_Stmt_for_end_expanded[10000]; // Element size 8
; CHECK: double MemRef_B_Stmt_for_end_expanded[10000]; // Element size 8
;
; Check that the memory accesses are modified except those related to tmp_04.
;
; CHECK-NOT: new: { Stmt_for_body[i0] -> MemRef_tmp_04__phi_Stmt_for_body_expanded[i0] };
; CHECK: new: { Stmt_for_body[i0] -> MemRef_tmp_11__phi_Stmt_for_inc_expanded[i0, 0] };
; CHECK: new: { Stmt_for_inc[i0, i1] -> MemRef_tmp_11__phi_Stmt_for_inc_expanded[i0, 1 + i1] : i1 <= 9998 };
; CHECK: new: { Stmt_for_inc[i0, i1] -> MemRef_tmp_11__phi_Stmt_for_inc_expanded[i0, i1] };
; CHECK: new: { Stmt_for_inc[i0, 9999] -> MemRef_add_lcssa__phi_Stmt_for_end_expanded[i0] };
; CHECK-NOT: new: { Stmt_for_end[i0] -> MemRef_tmp_04__phi_Stmt_for_body_expanded[1 + i0] : i0 <= 9998 };
; CHECK: new: { Stmt_for_end[i0] -> MemRef_add_lcssa__phi_Stmt_for_end_expanded[i0] };
; CHECK: new: { Stmt_for_end[i0] -> MemRef_B_Stmt_for_end_expanded[i0] };
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @tmp(double* %A, double* %B) {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %entry.split, %for.end
  %indvars.iv = phi i64 [ 0, %entry.split ], [ %indvars.iv.next, %for.end ]
  %tmp.04 = phi double [ 6.000000e+00, %entry.split ], [ %add.lcssa, %for.end ]
  br label %for.inc

for.inc:                                          ; preds = %for.body, %for.inc
  %j1.02 = phi i32 [ 0, %for.body ], [ %inc, %for.inc ]
  %tmp.11 = phi double [ %tmp.04, %for.body ], [ %add, %for.inc ]
  %add = fadd double %tmp.11, 2.000000e+00
  %inc = add nuw nsw i32 %j1.02, 1
  %exitcond = icmp ne i32 %inc, 10000
  br i1 %exitcond, label %for.inc, label %for.end

for.end:                                          ; preds = %for.inc
  %add.lcssa = phi double [ %add, %for.inc ]
  %arrayidx = getelementptr inbounds double, double* %B, i64 %indvars.iv
  store double %add.lcssa, double* %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond5 = icmp ne i64 %indvars.iv.next, 10000
  br i1 %exitcond5, label %for.body, label %for.end7

for.end7:                                         ; preds = %for.end
  ret void
}
