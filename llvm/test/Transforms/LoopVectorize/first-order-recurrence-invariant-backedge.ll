; RUN: opt -passes=loop-vectorize -force-vector-width=2 \
; RUN:   -force-vector-interleave=1 -S %s | FileCheck %s

; GVN load PRE can create a recurrence whose latch value is loop invariant.
; Make sure the recurrence is recognized and widened.

; CHECK-LABEL: define void @invariant_backedge_phi(
; CHECK: vector.ph:
; CHECK: [[SPLATINSERT:%.*]] = insertelement <2 x double> poison, double %next, i64 0
; CHECK: [[SPLAT:%.*]] = shufflevector <2 x double> [[SPLATINSERT]], <2 x double> poison, <2 x i32> zeroinitializer
; CHECK: [[RECURINIT:%.*]] = insertelement <2 x double> poison, double %init, i32 1
; CHECK: vector.body:
; CHECK: [[RECUR:%.*]] = phi <2 x double> [ [[RECURINIT]], %vector.ph ], [ [[SPLAT]], %vector.body ]
; CHECK: shufflevector <2 x double> [[RECUR]], <2 x double> [[SPLAT]], <2 x i32> <i32 1, i32 2>

define void @invariant_backedge_phi(
    ptr noalias %dst, ptr noalias readonly %src,
    ptr noalias readonly %coeffs, i64 %n) {
entry:
  %init = load double, ptr %coeffs, align 8
  %next.ptr = getelementptr inbounds double, ptr %coeffs, i64 1
  %next = load double, ptr %next.ptr, align 8
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %factor = phi double [ %init, %entry ], [ %next, %loop ]
  %src.ptr = getelementptr inbounds double, ptr %src, i64 %iv
  %x = load double, ptr %src.ptr, align 8
  %result = fmul double %factor, %x
  %dst.ptr = getelementptr inbounds double, ptr %dst, i64 %iv
  store double %result, ptr %dst.ptr, align 8
  %iv.next = add nuw i64 %iv, 1
  %done = icmp eq i64 %iv.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}
