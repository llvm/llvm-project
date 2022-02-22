; RUN: opt %loadPolly -polly-stmt-granularity=bb -polly-mse -analyze < %s | FileCheck %s
; RUN: opt %loadPolly -polly-stmt-granularity=bb -polly-mse -pass-remarks-analysis="polly-mse" -analyze < %s 2>&1 | FileCheck %s --check-prefix=MSE
;
; Verify that the accesses are correctly expanded for MemoryKind::PHI
; tmp_05 and tmp2_06 are not expanded because they need copy-in.
;
; Original source code :
;
; #define Ni 10000
; #define Nj 10000
;
; void mse(double A[Ni], double B[Nj]) {
;   int i,j;
;   double tmp = 6;
;   double tmp2 = 9;
;   for (i = 0; i < Ni; i++) {
;     for(j = 0; j < Nj; j++) {
;       tmp = tmp + tmp2;
;       tmp2 = i*j;
;     }
;   }
; }
;
; Check that the pass detects that tmp_05 and tmp2_06 read from their original values.
;
; MSE-DAG: MemRef_tmp_05__phi read from its original value.
; MSE-DAG: MemRef_tmp2_06__phi read from its original value.
;
; Check that the SAI are created except the expanded SAI of tmp_05 and tmp2_06.
;
; CHECK-DAG: double MemRef_add_lcssa__phi_Stmt_for_inc4_expanded[10000]; // Element size 8
; CHECK-DAG: double MemRef_tmp2_13__phi_Stmt_for_inc_expanded[10000][10000]; // Element size
; CHECK-DAG: double MemRef_conv_lcssa__phi_Stmt_for_inc4_expanded[10000]; // Element size 8
; CHECK-DAG: double MemRef_tmp_12__phi_Stmt_for_inc_expanded[10000][10000]; // Element size 8
; CHECK-NOT: double MemRef_tmp_05__phi_Stmt_for_body_expanded[10000]; // Element size 8
; CHECK-NOT: double MemRef_tmp2_06__phi_Stmt_for_body_expanded[10000]; // Element size 8
;
; Check that the memory accesses are modified except those related to tmp_05 and tmp_06.
;
; CHECK-NOT: new: { Stmt_for_body[i0] -> MemRef_tmp2_06__phi_Stmt_for_body_expanded[i0] };
; CHECK-NOT: new: { Stmt_for_body[i0] -> MemRef_tmp_05__phi_Stmt_for_body_expanded[i0] };
; CHECK: new: { Stmt_for_body[i0] -> MemRef_tmp2_13__phi_Stmt_for_inc_expanded[i0, 0] };
; CHECK: new: { Stmt_for_body[i0] -> MemRef_tmp_12__phi_Stmt_for_inc_expanded[i0, 0] };
; CHECK: new: { Stmt_for_inc[i0, i1] -> MemRef_tmp2_13__phi_Stmt_for_inc_expanded[i0, 1 + i1] : i1 <= 9998 };
; CHECK: new: { Stmt_for_inc[i0, i1] -> MemRef_tmp2_13__phi_Stmt_for_inc_expanded[i0, i1] };
; CHECK: new: { Stmt_for_inc[i0, i1] -> MemRef_tmp_12__phi_Stmt_for_inc_expanded[i0, 1 + i1] : i1 <= 9998 };
; CHECK: new: { Stmt_for_inc[i0, i1] -> MemRef_tmp_12__phi_Stmt_for_inc_expanded[i0, i1] };
; CHECK: new: { Stmt_for_inc[i0, 9999] -> MemRef_conv_lcssa__phi_Stmt_for_inc4_expanded[i0] };
; CHECK: new: { Stmt_for_inc[i0, 9999] -> MemRef_add_lcssa__phi_Stmt_for_inc4_expanded[i0] };
; CHECK-NOT: new: { Stmt_for_inc4[i0] -> MemRef_tmp2_06__phi_Stmt_for_body_expanded[1 + i0] : i0 <= 9998 };
; CHECK-NOT: new: { Stmt_for_inc4[i0] -> MemRef_tmp_05__phi_Stmt_for_body_expanded[1 + i0] : i0 <= 9998 };
; CHECK: new: { Stmt_for_inc4[i0] -> MemRef_conv_lcssa__phi_Stmt_for_inc4_expanded[i0] };
; CHECK: new: { Stmt_for_inc4[i0] -> MemRef_add_lcssa__phi_Stmt_for_inc4_expanded[i0] };
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @tmp(double* %A, double* %B) {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %entry.split, %for.inc4
  %tmp2.06 = phi double [ 9.000000e+00, %entry.split ], [ %conv.lcssa, %for.inc4 ]
  %tmp.05 = phi double [ 6.000000e+00, %entry.split ], [ %add.lcssa, %for.inc4 ]
  %i.04 = phi i32 [ 0, %entry.split ], [ %inc5, %for.inc4 ]
  br label %for.inc

for.inc:                                          ; preds = %for.body, %for.inc
  %tmp2.13 = phi double [ %tmp2.06, %for.body ], [ %conv, %for.inc ]
  %tmp.12 = phi double [ %tmp.05, %for.body ], [ %add, %for.inc ]
  %j.01 = phi i32 [ 0, %for.body ], [ %inc, %for.inc ]
  %mul = mul nuw nsw i32 %j.01, %i.04
  %conv = sitofp i32 %mul to double
  %add = fadd double %tmp.12, %tmp2.13
  %inc = add nuw nsw i32 %j.01, 1
  %exitcond = icmp ne i32 %inc, 10000
  br i1 %exitcond, label %for.inc, label %for.inc4

for.inc4:                                         ; preds = %for.inc
  %conv.lcssa = phi double [ %conv, %for.inc ]
  %add.lcssa = phi double [ %add, %for.inc ]
  %inc5 = add nuw nsw i32 %i.04, 1
  %exitcond7 = icmp ne i32 %inc5, 10000
  br i1 %exitcond7, label %for.body, label %for.end6

for.end6:                                         ; preds = %for.inc4
  ret void
}
