; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -disable-output < %s 2>&1 | FileCheck %s

;void f(long a[], long N, long M) {
;  long i, j, k;
;  for (j = 0; j < M; ++j)
;    if (true)
;      a[j] = j;
;    else {
;      a[j] = M;
;      a[j - N] = 0;
;    }
;}

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

define void @f(ptr nocapture %a, i64 %N, i64 %M) nounwind {
entry:
  %0 = icmp sgt i64 %M, 0                         ; <i1> [#uses=1]
  br i1 %0, label %bb, label %return

bb:                                               ; preds = %bb3, %entry
  %1 = phi i64 [ 0, %entry ], [ %2, %bb3 ]        ; <i64> [#uses=5]
  %scevgep = getelementptr i64, ptr %a, i64 %1        ; <ptr> [#uses=2]
  br i1 true, label %bb1, label %bb2

bb1:                                              ; preds = %bb
  store i64 %1, ptr %scevgep, align 8
  br label %bb3

bb2:                                              ; preds = %bb
  %tmp7 = sub i64 %1, %N                          ; <i64> [#uses=1]
  %scevgep8 = getelementptr i64, ptr %a, i64 %tmp7    ; <ptr> [#uses=1]
  store i64 %M, ptr %scevgep, align 8
  store i64 0, ptr %scevgep8, align 8
  br label %bb3

bb3:                                              ; preds = %bb2, %bb1
  %2 = add nsw i64 %1, 1                          ; <i64> [#uses=2]
  %exitcond = icmp eq i64 %2, %M                  ; <i1> [#uses=1]
  br i1 %exitcond, label %return, label %bb

return:                                           ; preds = %bb3, %entry
  ret void
}

; CHECK:     Stmt_bb1
; CHECK:       Domain :=
; CHECK:         [M] -> { Stmt_bb1[i0] : 0 <= i0 < M };
; CHECK-NOT: Stmt_bb2
