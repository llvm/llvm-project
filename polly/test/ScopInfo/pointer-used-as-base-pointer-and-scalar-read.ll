; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s

; In this test case we pass a pointer %A into a PHI node and also use this
; pointer as base pointer of an array store. As a result, we get both scalar
; and array memory accesses to A[] and A[0].

; CHECK:      Arrays {
; CHECK-NEXT:     float MemRef_A[*]; // Element size 4
; CHECK-NEXT:     float* MemRef_x__phi; // Element size 8
; CHECK-NEXT:     float* MemRef_C[*]; // Element size 8
; CHECK-NEXT: }
; CHECK:      Arrays (Bounds as pw_affs) {
; CHECK-NEXT:     float MemRef_A[*]; // Element size 4
; CHECK-NEXT:     float* MemRef_x__phi; // Element size 8
; CHECK-NEXT:     float* MemRef_C[*]; // Element size 8
; CHECK-NEXT: }
; CHECK:      Alias Groups (0):
; CHECK-NEXT:     n/a
; CHECK:      Statements {
; CHECK-NEXT:     Stmt_then
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [p] -> { Stmt_then[i0] : p = 32 and 0 <= i0 <= 999 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [p] -> { Stmt_then[i0] -> [i0, 1] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [p] -> { Stmt_then[i0] -> MemRef_A[0] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [p] -> { Stmt_then[i0] -> MemRef_x__phi[] };
; CHECK-NEXT:     Stmt_else
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [p] -> { Stmt_else[i0] : 0 <= i0 <= 999 and (p >= 33 or p <= 31) };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [p] -> { Stmt_else[i0] -> [i0, 0] : p >= 33 or p <= 31 };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [p] -> { Stmt_else[i0] -> MemRef_A[0] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [p] -> { Stmt_else[i0] -> MemRef_x__phi[] };
; CHECK-NEXT:     Stmt_bb8
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [p] -> { Stmt_bb8[i0] : 0 <= i0 <= 999 and (p >= 33 or p <= 32) };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [p] -> { Stmt_bb8[i0] -> [i0, 2] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [p] -> { Stmt_bb8[i0] -> MemRef_x__phi[] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [p] -> { Stmt_bb8[i0] -> MemRef_C[0] };
; CHECK-NEXT: }

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo(float* noalias %A, float* noalias %B, float ** noalias %C, i32 %p) {
bb:
  br label %bb1

bb1:
  %i.0 = phi i64 [ 0, %bb ], [ %tmp9, %bb8 ]
  %exitcond = icmp ne i64 %i.0, 1000
  br i1 %exitcond, label %bb2, label %bb10

bb2:
  %cmp = icmp eq i32 %p, 32
  br i1 %cmp, label %then, label %else

then:
  store float 3.0, float* %A
  br label %bb8

else:
  store float 4.0, float* %A
  br label %bb8

bb8:
  %x = phi float* [%A, %then], [%B, %else]
  store float* %x, float** %C
  %tmp9 = add nuw nsw i64 %i.0, 1
  br label %bb1

bb10:
  ret void
}
