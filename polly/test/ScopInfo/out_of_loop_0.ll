; RUN: opt %loadPolly %defaultOpts  -polly-analyze-ir  -analyze < %s | FileCheck %s
; RUN: opt %loadPolly %defaultOpts -polly-analyze-ir  -analyze < %s | FileCheck %s
; XFAIL: *
;void f(long a[], long N, long M) {
;  long i;

;  for (i = 0; i < N; ++i)
;   a[i] = i;

;  a[M] = 0;
;}

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

define void @f(i64* nocapture %a, i64 %N, i64 %M) nounwind {
entry:
  %0 = icmp sgt i64 %N, 0                         ; <i1> [#uses=1]
  br i1 %0, label %bb, label %bb2

bb:                                               ; preds = %bb, %entry
  %1 = phi i64 [ 0, %entry ], [ %2, %bb ]         ; <i64> [#uses=3]
  %scevgep = getelementptr i64* %a, i64 %1        ; <i64*> [#uses=1]
  store i64 %1, i64* %scevgep, align 8
  %2 = add nsw i64 %1, 1                          ; <i64> [#uses=2]
  %exitcond = icmp eq i64 %2, %N                  ; <i1> [#uses=1]
  br i1 %exitcond, label %bb2, label %bb

bb2:                                              ; preds = %bb, %entry
  %3 = getelementptr inbounds i64* %a, i64 %M     ; <i64*> [#uses=1]
  store i64 0, i64* %3, align 8
  ret void
}

; CHECK: Scop: entry => <Function Return>        Parameters: (%N, %M, )
