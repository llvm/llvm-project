; RUN: opt %loadPolly -polly-print-ast -disable-output < %s | FileCheck %s

; CHECK:      {
; CHECK-NEXT:    if (p <= -1 || p >= 1)
; CHECK-NEXT:      Stmt_preheader();
; CHECK-NEXT:    for (int c0 = 0; c0 < 2 * p; c0 += 1)
; CHECK-NEXT:      Stmt_loop(c0);
; CHECK-NEXT:    if (p <= -1) {
; CHECK-NEXT:      for (int c0 = 0; c0 <= 2 * p + 255; c0 += 1)
; CHECK-NEXT:        Stmt_loop(c0);
; CHECK-NEXT:    } else if (p == 0) {
; CHECK-NEXT:      Stmt_side();
; CHECK-NEXT:    }
; CHECK-NEXT:  }

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"

define void @zot(float* %A, i32 %arg) {
bb:
  %p = ashr i32 %arg, 25
  %tmpEven = shl nsw i32 %p, 1
  %tmp3 = and i32 %tmpEven, 254
  br label %cond

cond:
  %tmpEvenTrunc = trunc i32 %tmpEven to i8
  %br.cmp = icmp eq i8 %tmpEvenTrunc, 0
  br i1 %br.cmp, label %side, label %preheader

preheader:
  store float 1.0, float* %A
  br label %loop

loop:
  %indvar = phi i32 [ %indvar.next, %loop ], [ 1, %preheader ]
  store float 1.0, float* %A
  %indvar.next = add nuw nsw i32 %indvar, 1
  %cmp = icmp eq i32 %indvar, %tmp3
  br i1 %cmp, label %exit, label %loop

side:
  store float 1.0, float* %A
  br label %ret

exit:
  br label %ret

ret:
  ret void
}
