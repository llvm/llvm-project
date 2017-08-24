; RUN: opt %loadPolly -polly-mse -analyze < %s | FileCheck %s
; RUN: opt %loadPolly -polly-mse -pass-remarks-analysis="polly-mse" -analyze < %s 2>&1| FileCheck %s --check-prefix=MSE
;
; Verify that the accesses are correctly expanded for MemoryKind::Array and MemoryKind::PHI.
; tmp_06_phi is not expanded because it need copy in.
;
; Original source code :
;
; #define Ni 2000
; #define Nj 3000
; 
; void tmp3(double A[Ni], double B[Nj]) {
;   int i,j;
;   double tmp = 6;
;   for (i = 0; i < Ni; i++) {
; 
;     for(int h = 0; h<Nj; h++)
;       B[h] = h;
;     
;     for(j = 0; j < Nj; j++) {
;       for(int k=0; k<Nj; k++) {
; 	tmp = tmp+i+k+j;
; 	A[i+j] = tmp*B[k];
;       }
;     }
;   }
; }
;
; Check if the expanded SAI are created except for tmp_06_phi
;
; MSE: MemRef_tmp_06__phi read from its original value.
;
; CHECK-DAG: double MemRef_A_Stmt_for_body11_expanded[10000][10000][10000]; // Element size 8
; CHECK-DAG: double MemRef_add16_lcssa__phi_Stmt_for_inc25_expanded[10000][10000]; // Element size 8
; CHECK-DAG: double MemRef_B_Stmt_for_body3_expanded[10000][10000]; // Element size 8
; CHECK-DAG: double MemRef_tmp_06_Stmt_for_body_expanded[10000]; // Element size 8
; CHECK-DAG: double MemRef_add16_lcssa_lcssa__phi_Stmt_for_inc28_expanded[10000]; // Element size 8
; CHECK-DAG: double MemRef_tmp_14__phi_Stmt_for_body7_expanded[10000][10000]; // Element size 8
; CHECK-DAG: double MemRef_tmp_22__phi_Stmt_for_body11_expanded[10000][10000][10000]; // Element size 8
; CHECK-NOT: double MemRef_tmp_06__phi_Stmt_for_body_expanded[10000]; // Element size 8
;
; Check if the memory accesses are modified except those of tmp_06_phi
;
; CHECK-NOT: new: { Stmt_for_body[i0] -> MemRef_tmp_06__phi_Stmt_for_body_expanded[i0] };
; CHECK-DAG: new: { Stmt_for_body[i0] -> MemRef_tmp_06_Stmt_for_body_expanded[i0] };
; CHECK-DAG: new: { Stmt_for_body3[i0, i1] -> MemRef_B_Stmt_for_body3_expanded[i0, i1] };
; CHECK-DAG: new: { Stmt_for_end[i0] -> MemRef_tmp_06_Stmt_for_body_expanded[i0] };
; CHECK-DAG: new: { Stmt_for_end[i0] -> MemRef_tmp_14__phi_Stmt_for_body7_expanded[i0, 0] };
; CHECK-DAG: new: { Stmt_for_body7[i0, i1] -> MemRef_tmp_14__phi_Stmt_for_body7_expanded[i0, i1] };
; CHECK-DAG: new: { Stmt_for_body7[i0, i1] -> MemRef_tmp_22__phi_Stmt_for_body11_expanded[i0, i1, 0] };
; CHECK-DAG: new: { Stmt_for_body11[i0, i1, i2] -> MemRef_tmp_22__phi_Stmt_for_body11_expanded[i0, i1, 1 + i2] : i2 <= 9998 };
; CHECK-DAG: new: { Stmt_for_body11[i0, i1, i2] -> MemRef_tmp_22__phi_Stmt_for_body11_expanded[i0, i1, i2] };
; CHECK-DAG: new: { Stmt_for_body11[i0, i1, i2] -> MemRef_B_Stmt_for_body3_expanded[i0, i2] };
; CHECK-DAG: new: { Stmt_for_body11[i0, i1, i2] -> MemRef_A_Stmt_for_body11_expanded[i0, i1, i2] };
; CHECK-DAG: new: { Stmt_for_body11[i0, i1, 9999] -> MemRef_add16_lcssa__phi_Stmt_for_inc25_expanded[i0, i1] };
; CHECK-DAG: new: { Stmt_for_inc25[i0, i1] -> MemRef_tmp_14__phi_Stmt_for_body7_expanded[i0, 1 + i1] : i1 <= 9998 };
; CHECK-DAG: new: { Stmt_for_inc25[i0, i1] -> MemRef_add16_lcssa__phi_Stmt_for_inc25_expanded[i0, i1] };
; CHECK-DAG: new: { Stmt_for_inc25[i0, 9999] -> MemRef_add16_lcssa_lcssa__phi_Stmt_for_inc28_expanded[i0] };
; CHECK-DAG: new: { Stmt_for_inc28[i0] -> MemRef_add16_lcssa_lcssa__phi_Stmt_for_inc28_expanded[i0] };
; CHECK-NOT: new: { Stmt_for_inc28[i0] -> MemRef_tmp_06__phi_Stmt_for_body_expanded[1 + i0] : i0 <= 9998 };
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @mse(double* %A, double* %B) {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %entry.split, %for.inc28
  %indvars.iv15 = phi i64 [ 0, %entry.split ], [ %indvars.iv.next16, %for.inc28 ]
  %tmp.06 = phi double [ 6.000000e+00, %entry.split ], [ %add16.lcssa.lcssa, %for.inc28 ]
  br label %for.body3

for.body3:                                        ; preds = %for.body, %for.body3
  %indvars.iv = phi i64 [ 0, %for.body ], [ %indvars.iv.next, %for.body3 ]
  %0 = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %0 to double
  %arrayidx = getelementptr inbounds double, double* %B, i64 %indvars.iv
  store double %conv, double* %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 10000
  br i1 %exitcond, label %for.body3, label %for.end

for.end:                                          ; preds = %for.body3
  br label %for.body7

for.body7:                                        ; preds = %for.end, %for.inc25
  %indvars.iv11 = phi i64 [ 0, %for.end ], [ %indvars.iv.next12, %for.inc25 ]
  %tmp.14 = phi double [ %tmp.06, %for.end ], [ %add16.lcssa, %for.inc25 ]
  br label %for.body11

for.body11:                                       ; preds = %for.body7, %for.body11
  %indvars.iv8 = phi i64 [ 0, %for.body7 ], [ %indvars.iv.next9, %for.body11 ]
  %tmp.22 = phi double [ %tmp.14, %for.body7 ], [ %add16, %for.body11 ]
  %1 = trunc i64 %indvars.iv15 to i32
  %conv12 = sitofp i32 %1 to double
  %add = fadd double %tmp.22, %conv12
  %2 = trunc i64 %indvars.iv8 to i32
  %conv13 = sitofp i32 %2 to double
  %add14 = fadd double %add, %conv13
  %3 = trunc i64 %indvars.iv11 to i32
  %conv15 = sitofp i32 %3 to double
  %add16 = fadd double %add14, %conv15
  %arrayidx18 = getelementptr inbounds double, double* %B, i64 %indvars.iv8
  %4 = load double, double* %arrayidx18, align 8
  %mul = fmul double %add16, %4
  %5 = add nuw nsw i64 %indvars.iv11, %indvars.iv15
  %arrayidx21 = getelementptr inbounds double, double* %A, i64 %5
  store double %mul, double* %arrayidx21, align 8
  %indvars.iv.next9 = add nuw nsw i64 %indvars.iv8, 1
  %exitcond10 = icmp ne i64 %indvars.iv.next9, 10000
  br i1 %exitcond10, label %for.body11, label %for.inc25

for.inc25:                                        ; preds = %for.body11
  %add16.lcssa = phi double [ %add16, %for.body11 ]
  %indvars.iv.next12 = add nuw nsw i64 %indvars.iv11, 1
  %exitcond14 = icmp ne i64 %indvars.iv.next12, 10000
  br i1 %exitcond14, label %for.body7, label %for.inc28

for.inc28:                                        ; preds = %for.inc25
  %add16.lcssa.lcssa = phi double [ %add16.lcssa, %for.inc25 ]
  %indvars.iv.next16 = add nuw nsw i64 %indvars.iv15, 1
  %exitcond17 = icmp ne i64 %indvars.iv.next16, 10000
  br i1 %exitcond17, label %for.body, label %for.end30

for.end30:                                        ; preds = %for.inc28
  ret void
}
