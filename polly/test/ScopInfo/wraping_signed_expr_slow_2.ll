; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -disable-output < %s 2>&1 | FileCheck %s
;
; This checks that the no-wraps checks will be computed fast as some example
; already showed huge slowdowns even though the inbounds and nsw flags were
; all in place.
;
;    // Inspired by itrans8x8 in transform8x8.c from the ldecode benchmark.
;    void fast(char *A, char N, char M) {
;      for (char i = 0; i < 8; i++) {
;        char  index0 = i + N;
;        char  index1 = index0 * 16;
;        char  index2 = index1 + M;
;        A[(short)index2]++;
;      }
;    }
;
;    void slow(char *A, char N, char M) {
;      for (char i = 0; i < 8; i++) {
;        char  index0 = i + N;
;        char  index1 = index0 * 16;
;        short index2 = ((short)index1) + ((short)M);
;        A[index2]++;
;      }
;    }
;
; CHECK: Function: fast
; CHECK: Function: slow
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @fast(ptr %A, i8 %N, i8 %M) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i8 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i8 %indvars.iv, 8
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %tmp3 = add nsw i8 %indvars.iv, %N
  %mul = mul nsw i8 %tmp3, 16
  %add2 = add nsw i8 %mul, %M
  %add2ext = sext i8 %add2 to i16
  %arrayidx = getelementptr inbounds i8, ptr %A, i16 %add2ext
  %tmp4 = load i8, ptr %arrayidx, align 4
  %inc = add nsw i8 %tmp4, 1
  store i8 %inc, ptr %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i8 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

define void @slow(ptr %A, i8 %N, i8 %M) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i8 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i8 %indvars.iv, 8
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %tmp3 = add nsw i8 %indvars.iv, %N
  %mul = mul nsw i8 %tmp3, 16
  %mulext = sext i8 %mul to i16
  %Mext = sext i8 %M to i16
  %add2 = add nsw i16 %mulext, %Mext
  %arrayidx = getelementptr inbounds i8, ptr %A, i16 %add2
  %tmp4 = load i8, ptr %arrayidx, align 4
  %inc = add nsw i8 %tmp4, 1
  store i8 %inc, ptr %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i8 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
