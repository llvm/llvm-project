; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s
;
; This checks that the no-wraps checks will be computed fast as some example
; already showed huge slowdowns even though the inbounds and nsw flags were
; all in place.
;
;    // Inspired by itrans8x8 in transform8x8.c from the ldecode benchmark.
;    void fast(char *A, char N, char M) {
;      for (char i = 0; i < 8; i++) {
;        short index0 = (short)(i + N);
;        #ifdef fast
;          short index1 = (index0 *  1) + (short)M;
;        #else
;          short index1 = (index0 * 16) + (short)M;
;        #endif
;        A[index1]++;
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
  %tmp3ext = sext i8 %tmp3 to i16
  ;%mul = mul nsw i16 %tmp3ext, 16
  %Mext = sext i8 %M to i16
  %add2 = add nsw i16 %tmp3ext, %Mext
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

define void @slow(ptr %A, i8 %N, i8 %M) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i8 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i8 %indvars.iv, 8
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %tmp3 = add nsw i8 %indvars.iv, %N
  %tmp3ext = sext i8 %tmp3 to i16
  %mul = mul nsw i16 %tmp3ext, 16
  %Mext = sext i8 %M to i16
  %add2 = add nsw i16 %mul, %Mext
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
