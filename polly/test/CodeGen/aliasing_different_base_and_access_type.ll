; RUN: opt -opaque-pointers=0 %loadPolly -S -polly-codegen < %s | FileCheck %s
;
; We have to cast %B to "short *" before we create RTCs.
;
; CHECK:   %polly.access.cast.B = bitcast i32* %B to i16*
; CHECK-NEXT:   %polly.access.B = getelementptr i16, i16* %polly.access.cast.B, i64 1024
;
; We should never access %B as an i32 pointer:
;
; CHECK-NOT: getelementptr i32, i32* %B
;
;    void jd(int *A, int *B) {
;      for (int i = 0; i < 1024; i++)
;        A[i] = ((short *)B)[i];
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @jd(i32* %A, i32* %B) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i64 %indvars.iv, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %tmp = bitcast i32* %B to i16*
  %arrayidx = getelementptr inbounds i16, i16* %tmp, i64 %indvars.iv
  %tmp1 = load i16, i16* %arrayidx, align 2
  %conv = sext i16 %tmp1 to i32
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  store i32 %conv, i32* %arrayidx2, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
