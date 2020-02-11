; RUN: opt %loadPolly -polly-analysis-computeout=0 -polly-scops -polly-rtc-max-parameters=8 -analyze < %s | FileCheck %s --check-prefix=MAX8
; RUN: opt %loadPolly -polly-analysis-computeout=0 -polly-scops -polly-rtc-max-parameters=7 -analyze < %s | FileCheck %s --check-prefix=MAX7
;
; Check that we allow this SCoP even though it has 10 parameters involved in posisbly aliasing accesses.
; However, only 7 are involved in accesses through B, 8 through C and none in accesses through A.
;
; MAX8-LABEL:  Function: jd
; MAX8-NEXT: Region: %for.cond---%for.end

; MAX7:  Invalid Scop!
;
;    void jd(int *A, int *B, int *C, long p1, long p2, long p3, long p4, long p5,
;            long p6, long p7, long p8, long p9, long p10) {
;      for (int i = 0; i < 1024; i++)
;        A[i] = B[p1] - B[p2] + B[-p3] - B[p4] + B[p5] - B[-p6] + B[p7] - C[p3] +
;               C[-p4] - C[p5] + C[p6] - C[-p7] + C[p8] - C[p9] + C[-p10];
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @jd(i32* %A, i32* %B, i32* %C, i64 %p1, i64 %p2, i64 %p3, i64 %p4, i64 %p5, i64 %p6, i64 %p7, i64 %p8, i64 %p9, i64 %p10) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i64 %indvars.iv, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds i32, i32* %B, i64 %p1
  %tmp = load i32, i32* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %B, i64 %p2
  %tmp1 = load i32, i32* %arrayidx1, align 4
  %sub = sub nsw i32 %tmp, %tmp1
  %sub2 = sub nsw i64 0, %p3
  %arrayidx3 = getelementptr inbounds i32, i32* %B, i64 %sub2
  %tmp2 = load i32, i32* %arrayidx3, align 4
  %add = add nsw i32 %sub, %tmp2
  %arrayidx4 = getelementptr inbounds i32, i32* %B, i64 %p4
  %tmp3 = load i32, i32* %arrayidx4, align 4
  %sub5 = sub nsw i32 %add, %tmp3
  %arrayidx6 = getelementptr inbounds i32, i32* %B, i64 %p5
  %tmp4 = load i32, i32* %arrayidx6, align 4
  %add7 = add nsw i32 %sub5, %tmp4
  %sub8 = sub nsw i64 0, %p6
  %arrayidx9 = getelementptr inbounds i32, i32* %B, i64 %sub8
  %tmp5 = load i32, i32* %arrayidx9, align 4
  %sub10 = sub nsw i32 %add7, %tmp5
  %arrayidx11 = getelementptr inbounds i32, i32* %B, i64 %p7
  %tmp6 = load i32, i32* %arrayidx11, align 4
  %add12 = add nsw i32 %sub10, %tmp6
  %arrayidx13 = getelementptr inbounds i32, i32* %C, i64 %p3
  %tmp7 = load i32, i32* %arrayidx13, align 4
  %sub14 = sub nsw i32 %add12, %tmp7
  %sub15 = sub nsw i64 0, %p4
  %arrayidx16 = getelementptr inbounds i32, i32* %C, i64 %sub15
  %tmp8 = load i32, i32* %arrayidx16, align 4
  %add17 = add nsw i32 %sub14, %tmp8
  %arrayidx18 = getelementptr inbounds i32, i32* %C, i64 %p5
  %tmp9 = load i32, i32* %arrayidx18, align 4
  %sub19 = sub nsw i32 %add17, %tmp9
  %arrayidx20 = getelementptr inbounds i32, i32* %C, i64 %p6
  %tmp10 = load i32, i32* %arrayidx20, align 4
  %add21 = add nsw i32 %sub19, %tmp10
  %sub22 = sub nsw i64 0, %p7
  %arrayidx23 = getelementptr inbounds i32, i32* %C, i64 %sub22
  %tmp11 = load i32, i32* %arrayidx23, align 4
  %sub24 = sub nsw i32 %add21, %tmp11
  %arrayidx25 = getelementptr inbounds i32, i32* %C, i64 %p8
  %tmp12 = load i32, i32* %arrayidx25, align 4
  %add26 = add nsw i32 %sub24, %tmp12
  %arrayidx27 = getelementptr inbounds i32, i32* %C, i64 %p9
  %tmp13 = load i32, i32* %arrayidx27, align 4
  %sub28 = sub nsw i32 %add26, %tmp13
  %sub29 = sub nsw i64 0, %p10
  %arrayidx30 = getelementptr inbounds i32, i32* %C, i64 %sub29
  %tmp14 = load i32, i32* %arrayidx30, align 4
  %add31 = add nsw i32 %sub28, %tmp14
  %arrayidx32 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  store i32 %add31, i32* %arrayidx32, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
