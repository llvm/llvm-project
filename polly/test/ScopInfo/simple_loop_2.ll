; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s

; void f(int a[], int N) {
;   int i;
;   for (i = 0; i < N; ++i)
;     a[i] = i;
; }

; CHECK:      Assumed Context:
; CHECK-NEXT: [N] -> {  :  }
;
; CHECK:      Arrays {
; CHECK-NEXT:     i32 MemRef_a[*]; // Element size 4
; CHECK-NEXT: }
;
; CHECK:      Statements {
; CHECK-NEXT:     Stmt_bb
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [N] -> { Stmt_bb[i0] : 0 <= i0 < N };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [N] -> { Stmt_bb[i0] -> [i0] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [N] -> { Stmt_bb[i0] -> MemRef_a[i0] };
; CHECK-NEXT: }

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

define void @f(ptr nocapture %a, i64 %N) nounwind {
entry:
  br label %bb

bb:                                               ; preds = %bb, %entry
  %i = phi i32 [ 0, %entry ], [ %i.inc, %bb ]
  %scevgep = getelementptr inbounds i32, ptr %a, i32 %i
  store i32 %i, ptr %scevgep
  %i.inc = add nsw i32 %i, 1
  %i.ext = zext i32 %i.inc to i64
  %exitcond = icmp eq i64 %i.ext, %N
  br i1 %exitcond, label %return, label %bb

return:                                           ; preds = %bb, %entry
  ret void
}
