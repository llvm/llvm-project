; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -disable-output < %s 2>&1 | FileCheck %s

; CHECK:      Assumed Context:
; CHECK-NEXT: [N] -> {  :  }
; CHECK-NEXT: Invalid Context:
; CHECK-NEXT: [N] -> {  : N <= 0 or N >= 1152921504606846976 }
;
; CHECK:              Domain :=
; CHECK-NEXT:             [N] -> { Stmt_bb[i0] : 0 <= i0 < N; Stmt_bb[0] : N <= 0 };
;
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

define void @f(ptr nocapture %a, i64 %N) nounwind {
entry:
  br label %bb

bb:                                               ; preds = %bb, %entry
  %i = phi i64 [ %N, %entry ], [ %i.dec, %bb ]
  %scevgep = getelementptr inbounds i64, ptr %a, i64 %i
  store i64 %i, ptr %scevgep
  %i.dec = add nsw i64 %i, -1
  %exitcond = icmp ugt i64 %i.dec, 0
  br i1 %exitcond, label %bb, label %return

return:                                           ; preds = %bb, %entry
  ret void
}
