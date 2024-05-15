; RUN: opt %loadPolly -tbaa -polly-print-scops -polly-invariant-load-hoisting=true -polly-ignore-aliasing \
; RUN:                -disable-output < %s | FileCheck %s
;
; Note: The order of the invariant accesses is important because A is the
;       base pointer of tmp3 and we will generate code in the same order as
;       the invariant accesses are listed here.
;
; CHECK: Invariant Accesses: {
; CHECK:    ReadAccess := [Reduction Type: NONE] [Scalar: 0]
; CHECK:         MemRef_A[42]
; CHECK:    ReadAccess := [Reduction Type: NONE] [Scalar: 0]
; CHECK:         MemRef_tmp3[32]
; CHECK: }
;
; CHECK: Arrays {
; CHECK:   ptr MemRef_A[*];
; CHECK:   ptr MemRef_tmp3[*]; [BasePtrOrigin: MemRef_A]
; CHECK:   i32 MemRef_tmp5[*]; [BasePtrOrigin: MemRef_tmp3]
; CHECK: }
;
; CHECK: Arrays (Bounds as pw_affs) {
; CHECK:   ptr MemRef_A[*];
; CHECK:   ptr MemRef_tmp3[*]; [BasePtrOrigin: MemRef_A]
; CHECK:   i32 MemRef_tmp5[*]; [BasePtrOrigin: MemRef_tmp3]
; CHECK: }
;
;    void f(int ***A) {
;      for (int i = 0; i < 1024; i++)
;        A[42][32][i] = 0;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(ptr %A) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb7, %bb
  %indvars.iv = phi i64 [ %indvars.iv.next, %bb7 ], [ 0, %bb ]
  %exitcond = icmp ne i64 %indvars.iv, 1024
  br i1 %exitcond, label %bb2, label %bb8

bb2:                                              ; preds = %bb1
  %tmp = getelementptr inbounds ptr, ptr %A, i64 42
  %tmp3 = load ptr, ptr %tmp, align 8
  %tmp4 = getelementptr inbounds ptr, ptr %tmp3, i64 32
  %tmp5 = load ptr, ptr %tmp4, align 8
  %tmp6 = getelementptr inbounds i32, ptr %tmp5, i64 %indvars.iv
  store i32 0, ptr %tmp6, align 4
  br label %bb7

bb7:                                              ; preds = %bb2
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %bb1

bb8:                                              ; preds = %bb1
  ret void
}
