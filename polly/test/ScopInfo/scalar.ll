; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

define void @f(ptr %a, i64 %N) {
entry:
  br label %for

for:
  %indvar = phi i64 [ 0, %entry ], [ %indvar.next, %for.backedge ]
  br label %S1

S1:
  %scevgep1 = getelementptr i64, ptr %a, i64 %indvar
  %val = load i64, ptr %scevgep1, align 8
  br label %S2

S2:
  %scevgep2 = getelementptr i64, ptr %a, i64 %indvar
  store i64 %val, ptr %scevgep2, align 8
  br label %for.backedge

for.backedge:
  %indvar.next = add nsw i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, %N
  br i1 %exitcond, label %return, label %for

return:
  ret void
}


; CHECK:      Arrays {
; CHECK-NEXT:     i64 MemRef_a[*]; // Element size 8
; CHECK-NEXT:     i64 MemRef_val; // Element size 8
; CHECK-NEXT: }
;
; CHECK:      Arrays (Bounds as pw_affs) {
; CHECK-NEXT:     i64 MemRef_a[*]; // Element size 8
; CHECK-NEXT:     i64 MemRef_val; // Element size 8
; CHECK-NEXT: }
;
; CHECK:      Statements {
; CHECK-NEXT:     Stmt_S1
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [N] -> { Stmt_S1[i0] : 0 <= i0 < N };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [N] -> { Stmt_S1[i0] -> [i0, 0] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [N] -> { Stmt_S1[i0] -> MemRef_a[i0] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [N] -> { Stmt_S1[i0] -> MemRef_val[] };
; CHECK-NEXT:     Stmt_S2
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [N] -> { Stmt_S2[i0] : 0 <= i0 < N };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [N] -> { Stmt_S2[i0] -> [i0, 1] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [N] -> { Stmt_S2[i0] -> MemRef_a[i0] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [N] -> { Stmt_S2[i0] -> MemRef_val[] };
; CHECK-NEXT: }
