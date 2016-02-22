; RUN: opt %loadPolly -polly-allow-differing-element-types -polly-scops -analyze < %s | FileCheck %s
; RUN: opt %loadPolly -S -polly-allow-differing-element-types -polly-codegen < %s | FileCheck --check-prefix=IR %s
;
; CHECK:         Arrays {
; CHECK-NEXT:        i8 MemRef_A[*]; // Element size 1
; CHECK-NEXT:    }
; CHECK:         Statements {
; CHECK-NEXT:       Stmt_for_body3
; CHECK-NEXT:            Domain :=
; CHECK-NEXT:                { Stmt_for_body3[i0, i1] : 0 <= i0 <= 1023 and 0 <= i1 <= 1023 };
; CHECK-NEXT:            Schedule :=
; CHECK-NEXT:                { Stmt_for_body3[i0, i1] -> [i0, i1] };
; CHECK-NEXT:            MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                { Stmt_for_body3[i0, i1] -> MemRef_A[o0] : 0 <= o0 <= 186 };
;
; IR:   %[[r1:[a-zA-Z0-9]*]] = bitcast i32* %A to i8*
;
; IR: polly.stmt.for.body3:
; IR:   call void @llvm.memset.p0i8.i64(i8* %[[r1]], i8 36, i64 187, i32 4, i1 false)
;
;    #include <string.h>
;
;    void jd(int *A) {
;      for (int i = 0; i < 1024; i++)
;        for (int j = 0; j < 1024; j++)
;          memset(A, '$', 187);
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @jd(i32* noalias %A) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc4, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc5, %for.inc4 ]
  %exitcond1 = icmp ne i32 %i.0, 1024
  br i1 %exitcond1, label %for.body, label %for.end6

for.body:                                         ; preds = %for.cond
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %j.0 = phi i32 [ 0, %for.body ], [ %inc, %for.inc ]
  %exitcond = icmp ne i32 %j.0, 1024
  br i1 %exitcond, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  %tmp = bitcast i32* %A to i8*
  call void @llvm.memset.p0i8.i64(i8* %tmp, i8 36, i64 187, i32 4, i1 false)
  br label %for.inc

for.inc:                                          ; preds = %for.body3
  %inc = add nsw i32 %j.0, 1
  br label %for.cond1

for.end:                                          ; preds = %for.cond1
  br label %for.inc4

for.inc4:                                         ; preds = %for.end
  %inc5 = add nsw i32 %i.0, 1
  br label %for.cond

for.end6:                                         ; preds = %for.cond
  ret void
}

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1) #1

