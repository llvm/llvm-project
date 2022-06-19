; RUN: opt %loadPolly -polly-codegen -polly-allow-nonaffine-loops -polly-allow-nonaffine -debug-only=polly-dependence < %s 2>&1 | FileCheck %s
; REQUIRES: asserts

; CHECK:        MayWriteAccess :=   [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:       { Stmt_for_body__TO__for_inc11[i0] -> MemRef_A[o0] : 0 <= o0 <= 699 };
; CHECK-NEXT:   MayWriteAccess :=   [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:       { Stmt_for_body__TO__for_inc11[i0] -> MemRef_B[700] };

; The if condition C[i] is a non-affine condition, which make the nested loop boxed. The memory access for A should be a range A[0...699]. The memory access for B should be simplified to B[700].
;
; int A[1000], B[1000], C[1000];
;
; void foo(int n, int m, int N) {
;   for (int i = 0; i < 500; i+=1) { /* affine loop */
;      C[i] += i;
;      if (C[i]) { /* non-affine subregion */
;          int j;
;          for (j = 0; j < 700; j+=1) { /* boxed loop */
;            A[j] = 1;
;          }
;          B[j] = 2;
;      }
;    }
; }


target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"

@C = common global [1000 x i32] zeroinitializer, align 4
@A = common global [1000 x i32] zeroinitializer, align 4
@B = common global [1000 x i32] zeroinitializer, align 4

; Function Attrs: norecurse nounwind
define void @foo(i32 %n, i32 %m, i32 %N) #0 {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.inc11
  ret void

for.body:                                         ; preds = %for.inc11, %entry.split
  %indvars.iv25 = phi i64 [ 0, %entry.split ], [ %indvars.iv.next26, %for.inc11 ]
  %arrayidx = getelementptr inbounds [1000 x i32], [1000 x i32]* @C, i64 0, i64 %indvars.iv25
  %0 = load i32, i32* %arrayidx, align 4
  %1 = trunc i64 %indvars.iv25 to i32
  %add = add nsw i32 %0, %1
  store i32 %add, i32* %arrayidx, align 4
  %tobool = icmp eq i32 %add, 0
  br i1 %tobool, label %for.inc11, label %for.body5.preheader

for.body5.preheader:                              ; preds = %for.body
  br label %for.body5

for.body5:                                        ; preds = %for.body5.preheader, %for.body5
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body5 ], [ 0, %for.body5.preheader ]
  %arrayidx7 = getelementptr inbounds [1000 x i32], [1000 x i32]* @A, i64 0, i64 %indvars.iv
  store i32 1, i32* %arrayidx7, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv, 699
  br i1 %exitcond, label %for.end, label %for.body5

for.end:                                          ; preds = %for.body5
  store i32 2, i32* getelementptr inbounds ([1000 x i32], [1000 x i32]* @B, i64 0, i64 700), align 4
  br label %for.inc11

for.inc11:                                        ; preds = %for.body, %for.end
  %indvars.iv.next26 = add nuw nsw i64 %indvars.iv25, 1
  %exitcond27 = icmp eq i64 %indvars.iv25, 499
  br i1 %exitcond27, label %for.cond.cleanup, label %for.body
}

