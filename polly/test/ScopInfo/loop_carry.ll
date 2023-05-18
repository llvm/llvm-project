; RUN: opt %loadPolly -basic-aa -polly-stmt-granularity=bb -polly-print-scops -disable-output < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

;long f(long a[], long n) {
;  long i, k;
;  k = 1;
;  for (i = 1; i < n; ++i) {
;   a[i] = k * a[i - 1];
;   k = a[i + 3] + a[2 * i];
;  }
;  return 0;
;}

define i64 @f(ptr nocapture %a, i64 %n) nounwind {
entry:
  %0 = icmp sgt i64 %n, 1                         ; <i1> [#uses=1]
  br i1 %0, label %bb.nph, label %bb2

bb.nph:                                           ; preds = %entry
  %tmp = add i64 %n, -1                           ; <i64> [#uses=1]
  %.pre = load i64, ptr %a, align 8                   ; <i64> [#uses=1]
  br label %bb

bb:                                               ; preds = %bb, %bb.nph
  %tmp1 = phi i64 [ %.pre, %bb.nph ], [ %tmp2, %bb ]    ; <i64> [#uses=1]
  %indvar = phi i64 [ 0, %bb.nph ], [ %tmp6, %bb ] ; <i64> [#uses=3]
  %k.05 = phi i64 [ 1, %bb.nph ], [ %tmp5, %bb ]     ; <i64> [#uses=1]
  %tmp6 = add i64 %indvar, 1                      ; <i64> [#uses=3]
  %scevgep = getelementptr i64, ptr %a, i64 %tmp6     ; <ptr> [#uses=1]
  %tmp2 = mul nsw i64 %tmp1, %k.05                      ; <i64> [#uses=2]
  store i64 %tmp2, ptr %scevgep, align 8
  %tmp7 = shl i64 %indvar, 1                      ; <i64> [#uses=1]
  %tmp11 = add i64 %indvar, 4                     ; <i64> [#uses=1]
  %tmp8 = add i64 %tmp7, 2                        ; <i64> [#uses=1]
  %scevgep12 = getelementptr i64, ptr %a, i64 %tmp11  ; <ptr> [#uses=1]
  %scevgep9 = getelementptr i64, ptr %a, i64 %tmp8    ; <ptr> [#uses=1]
  %tmp3 = load i64, ptr %scevgep9, align 8               ; <i64> [#uses=1]
  %tmp4 = load i64, ptr %scevgep12, align 8              ; <i64> [#uses=1]
  %tmp5 = add nsw i64 %tmp3, %tmp4                         ; <i64> [#uses=1]
  %exitcond = icmp eq i64 %tmp6, %tmp             ; <i1> [#uses=1]
  br i1 %exitcond, label %bb2, label %bb

bb2:                                              ; preds = %bb, %entry
  ret i64 0
}


; CHECK:      Statements {
; CHECK-NEXT:     Stmt_bb
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [n] -> { Stmt_bb[i0] : 0 <= i0 <= -2 + n };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [n] -> { Stmt_bb[i0] -> [i0] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [n] -> { Stmt_bb[i0] -> MemRef_tmp1__phi[] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [n] -> { Stmt_bb[i0] -> MemRef_tmp1__phi[] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [n] -> { Stmt_bb[i0] -> MemRef_k_05__phi[] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [n] -> { Stmt_bb[i0] -> MemRef_k_05__phi[] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [n] -> { Stmt_bb[i0] -> MemRef_a[1 + i0] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [n] -> { Stmt_bb[i0] -> MemRef_a[2 + 2i0] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [n] -> { Stmt_bb[i0] -> MemRef_a[4 + i0] };
; CHECK-NEXT: }
