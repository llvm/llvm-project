; RUN: opt < %s -loop-reduce -S | FileCheck %s

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"

; for (int i = 0; i < n; ++i) {
;   use(i * 5 + 3);
;   // i * a + b is ephemeral and shouldn't be promoted by LSR
;   __builtin_assume(i * a + b >= 0);
; }
define void @ephemeral(i32 %a, i32 %b, i32 %n) {
; CHECK-LABEL: @ephemeral(
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %inc, %loop ]
  ; Only i and i * 5 + 3 should be indvars, not i * a + b.
; CHECK: phi i32
; CHECK: phi i32
; CHECK-NOT: phi i32
  %inc = add nsw i32 %i, 1
  %exitcond = icmp eq i32 %inc, %n

  %0 = mul nsw i32 %i, 5
  %1 = add nsw i32 %0, 3
  call void @use(i32 %1)

  %2 = mul nsw i32 %i, %a
  %3 = add nsw i32 %2, %b
  %4 = icmp sgt i32 %3, -1
  call void @llvm.assume(i1 %4)

  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

declare void @use(i32)

declare void @llvm.assume(i1)
