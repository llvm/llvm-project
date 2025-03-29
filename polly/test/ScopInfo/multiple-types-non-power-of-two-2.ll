; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -polly-allow-differing-element-types -disable-output < %s 2>&1 | FileCheck %s
;
;  void multiple_types(i8 *A) {
;    for (long i = 0; i < 100; i++) {
;      A[i] = *(i128 *)&A[16 * i] +
;             *(i192 *)&A[24 * i];
;    }
;  }
;
;
; CHECK: Arrays {
; CHECK:     i64 MemRef_A[*]; // Element size 8
; CHECK: }
; CHECK: Arrays (Bounds as pw_affs) {
; CHECK:     i64 MemRef_A[*]; // Element size 8
; CHECK: }
; CHECK: Alias Groups (0):
; CHECK:     n/a
; CHECK: Statements {
; CHECK:   Stmt_bb2
; CHECK:         Domain :=
; CHECK:             { Stmt_bb2[i0] : 0 <= i0 <= 99 };
; CHECK:         Schedule :=
; CHECK:             { Stmt_bb2[i0] -> [i0] };
; CHECK:         ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK:             { Stmt_bb2[i0] -> MemRef_A[o0] : 2i0 <= o0 <= 1 + 2i0 }
; CHECK:         ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK:             { Stmt_bb2[i0] -> MemRef_A[o0] : 3i0 <= o0 <= 2 + 3i0 }
; CHECK:         MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK:             { Stmt_bb2[i0] -> MemRef_A[o0] : 2i0 <= o0 <= 1 + 2i0 }
; CHECK: }

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @multiple_types(ptr %A) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb20, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp21, %bb20 ]
  %exitcond = icmp ne i64 %i.0, 100
  br i1 %exitcond, label %bb2, label %bb22

bb2:                                              ; preds = %bb1
  %load.i128.offset = mul i64 %i.0, 16
  %load.i128.ptr = getelementptr inbounds i8, ptr %A, i64 %load.i128.offset
  %load.i128.val = load i128, ptr %load.i128.ptr

  %load.i192.offset = mul i64 %i.0, 24
  %load.i192.ptr = getelementptr inbounds i8, ptr %A, i64 %load.i192.offset
  %load.i192.val = load i192, ptr %load.i192.ptr
  %load.i192.val.trunc = trunc i192 %load.i192.val to i128

  %sum = add i128 %load.i128.val, %load.i192.val.trunc
  store i128 %sum, ptr %load.i128.ptr
  br label %bb20

bb20:                                             ; preds = %bb2
  %tmp21 = add nuw nsw i64 %i.0, 1
  br label %bb1

bb22:                                             ; preds = %bb1
  ret void
}

