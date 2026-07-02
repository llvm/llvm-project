; RUN: opt < %s -passes='function(indvars,simplifycfg)' -S | FileCheck %s

declare void @llvm.trap()

; Loop bounded by i u< (count u/ 4); per-iter check `i*4 + 4 u> count` is dead.
; CHECK-LABEL: @udiv_mul_trap_elim
; CHECK-NOT:    call void @llvm.trap()
define void @udiv_mul_trap_elim(ptr %p, i64 %count) {
entry:
  %n = udiv i64 %count, 4
  %z = icmp eq i64 %n, 0
  br i1 %z, label %exit, label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inc, %latch ]
  %mul = mul nuw i64 %i, 4
  %end = add nuw i64 %mul, 4
  %ovf = icmp ugt i64 %end, %count
  br i1 %ovf, label %trap, label %latch

trap:
  call void @llvm.trap()
  unreachable

latch:
  %inc = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %inc, %n
  br i1 %c, label %loop, label %exit

exit:
  ret void
}

; Signed shape must NOT eliminate the trap (lemma is unsigned-only).
; CHECK-LABEL: @signed_keep_trap
; CHECK:        call void @llvm.trap()
define void @signed_keep_trap(ptr %p, i64 %count) {
entry:
  %n = sdiv i64 %count, 4
  %z = icmp sle i64 %n, 0
  br i1 %z, label %exit, label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inc, %latch ]
  %mul = mul nsw i64 %i, 4
  %end = add nsw i64 %mul, 4
  %ovf = icmp sgt i64 %end, %count
  br i1 %ovf, label %trap, label %latch

trap:
  call void @llvm.trap()
  unreachable

latch:
  %inc = add nsw i64 %i, 1
  %c = icmp slt i64 %inc, %n
  br i1 %c, label %loop, label %exit

exit:
  ret void
}
