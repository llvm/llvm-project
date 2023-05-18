; RUN: opt %loadPolly -polly-mse -polly-print-scops -disable-output < %s | FileCheck %s
; RUN: opt %loadNPMPolly "-passes=scop(print<polly-mse>)" -disable-output < %s | FileCheck %s
;
; Verify that the accesses are correctly expanded for MemoryKind::Array
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
;      B[h] = h;
;
;     for(j = 0; j < Nj; j++) {
;      for(int k=0; k<Nj; k++) {
; 	tmp = i+k+j;
; 	A[i+j] = tmp*B[k];
;       }
;     }
;   }
; }
;
; Check if the expanded SAI are created
;
; CHECK: double MemRef_B_Stmt_for_body3_expanded[10000][10000]; // Element size 8
; CHECK: double MemRef_A_Stmt_for_body11_expanded[10000][10000][10000]; // Element size 8
;
; Check if the memory accesses are modified
;
; CHECK: new: { Stmt_for_body3[i0, i1] -> MemRef_B_Stmt_for_body3_expanded[i0, i1] };
; CHECK: new: { Stmt_for_body11[i0, i1, i2] -> MemRef_B_Stmt_for_body3_expanded[i0, i2] };
; CHECK: new: { Stmt_for_body11[i0, i1, i2] -> MemRef_A_Stmt_for_body11_expanded[i0, i1, i2] };
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @mse(ptr %A, ptr %B) {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %entry.split, %for.inc25
  %indvars.iv14 = phi i64 [ 0, %entry.split ], [ %indvars.iv.next15, %for.inc25 ]
  br label %for.body3

for.body3:                                        ; preds = %for.body, %for.body3
  %indvars.iv = phi i64 [ 0, %for.body ], [ %indvars.iv.next, %for.body3 ]
  %0 = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %0 to double
  %arrayidx = getelementptr inbounds double, ptr %B, i64 %indvars.iv
  store double %conv, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 10000
  br i1 %exitcond, label %for.body3, label %for.end

for.end:                                          ; preds = %for.body3
  br label %for.body7

for.body7:                                        ; preds = %for.end, %for.inc22
  %indvars.iv9 = phi i64 [ 0, %for.end ], [ %indvars.iv.next10, %for.inc22 ]
  br label %for.body11

for.body11:                                       ; preds = %for.body7, %for.body11
  %indvars.iv5 = phi i64 [ 0, %for.body7 ], [ %indvars.iv.next6, %for.body11 ]
  %1 = add nuw nsw i64 %indvars.iv9, %indvars.iv14
  %2 = add nuw nsw i64 %1, %indvars.iv5
  %3 = trunc i64 %2 to i32
  %conv13 = sitofp i32 %3 to double
  %arrayidx15 = getelementptr inbounds double, ptr %B, i64 %indvars.iv5
  %4 = load double, ptr %arrayidx15, align 8
  %mul = fmul double %4, %conv13
  %5 = add nuw nsw i64 %indvars.iv9, %indvars.iv14
  %arrayidx18 = getelementptr inbounds double, ptr %A, i64 %5
  store double %mul, ptr %arrayidx18, align 8
  %indvars.iv.next6 = add nuw nsw i64 %indvars.iv5, 1
  %exitcond8 = icmp ne i64 %indvars.iv.next6, 10000
  br i1 %exitcond8, label %for.body11, label %for.inc22

for.inc22:                                        ; preds = %for.body11
  %indvars.iv.next10 = add nuw nsw i64 %indvars.iv9, 1
  %exitcond13 = icmp ne i64 %indvars.iv.next10, 10000
  br i1 %exitcond13, label %for.body7, label %for.inc25

for.inc25:                                        ; preds = %for.inc22
  %indvars.iv.next15 = add nuw nsw i64 %indvars.iv14, 1
  %exitcond16 = icmp ne i64 %indvars.iv.next15, 10000
  br i1 %exitcond16, label %for.body, label %for.end27

for.end27:                                        ; preds = %for.inc25
  ret void
}
