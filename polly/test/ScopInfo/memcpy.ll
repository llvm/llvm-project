; RUN: opt %loadPolly -basic-aa -polly-allow-differing-element-types -polly-print-scops -disable-output < %s | FileCheck %s
; RUN: opt %loadPolly -S -basic-aa -polly-allow-differing-element-types -polly-codegen < %s | FileCheck --check-prefix=IR %s
;
; CHECK:         Arrays {
; CHECK-NEXT:        i8 MemRef_A[*]; // Element size 1
; CHECK-NEXT:        i8 MemRef_B[*]; // Element size 1
; CHECK-NEXT:    }
; CHECK:         Statements {
; CHECK-NEXT:       Stmt_for_body3
; CHECK-NEXT:            Domain :=
; CHECK-NEXT:                { Stmt_for_body3[i0, i1] : 0 <= i0 <= 1023 and 0 <= i1 <= 1023 };
; CHECK-NEXT:            Schedule :=
; CHECK-NEXT:                { Stmt_for_body3[i0, i1] -> [i0, i1] };
; CHECK-NEXT:            MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                { Stmt_for_body3[i0, i1] -> MemRef_A[o0] : -16 <= o0 <= 20 };
; CHECK-NEXT:            ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                { Stmt_for_body3[i0, i1] -> MemRef_B[o0] : 64 <= o0 <= 100 };
;
; IR: polly.loop_preheader:
; IR:   %[[r1:[a-zA-Z0-9]*]] = getelementptr i32, i32* %A, i64 -4
; IR:   %[[r2:[a-zA-Z0-9]*]] = bitcast i32* %scevgep to i8*
; IR:   %[[r3:[a-zA-Z0-9]*]] = getelementptr i64, i64* %B, i64 8
; IR:   %[[r4:[a-zA-Z0-9]*]] = bitcast i64* %scevgep8 to i8*
;
; IR: polly.stmt.for.body3:
; IR:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %[[r2]], i8* align 4 %[[r4]], i64 37, i1 false)
;
;
;    #include <string.h>
;
;    void jd(int *restrict A, long *restrict B) {
;      for (int i = 0; i < 1024; i++)
;        for (int j = 0; j < 1024; j++)
;          memcpy(A - 4, B + 8, 37);
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @jd(i32* noalias %A, i64* noalias %B) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc5, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc6, %for.inc5 ]
  %exitcond1 = icmp ne i32 %i.0, 1024
  br i1 %exitcond1, label %for.body, label %for.end7

for.body:                                         ; preds = %for.cond
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %j.0 = phi i32 [ 0, %for.body ], [ %inc, %for.inc ]
  %exitcond = icmp ne i32 %j.0, 1024
  br i1 %exitcond, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  %add.ptr = getelementptr inbounds i32, i32* %A, i64 -4
  %tmp = bitcast i32* %add.ptr to i8*
  %add.ptr4 = getelementptr inbounds i64, i64* %B, i64 8
  %tmp2 = bitcast i64* %add.ptr4 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %tmp, i8* %tmp2, i64 37, i32 4, i1 false)
  br label %for.inc

for.inc:                                          ; preds = %for.body3
  %inc = add nsw i32 %j.0, 1
  br label %for.cond1

for.end:                                          ; preds = %for.cond1
  br label %for.inc5

for.inc5:                                         ; preds = %for.end
  %inc6 = add nsw i32 %i.0, 1
  br label %for.cond

for.end7:                                         ; preds = %for.cond
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i32, i1) #1

