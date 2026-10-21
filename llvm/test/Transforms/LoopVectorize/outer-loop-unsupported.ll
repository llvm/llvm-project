; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -enable-vplan-native-path -force-vector-width=4 -debug-only=loop-vectorize -disable-output %s 2>&1 | FileCheck %s

; Outer loops with constructs not supported by the VPlan-native path must be
; rejected.

; The preheader is terminated by an indirectbr, which cannot be replaced when
; creating the vector loop.
define void @outer_loop_preheader_indirectbr(i64 %n) {
; CHECK-LABEL: LV: Checking a loop in 'outer_loop_preheader_indirectbr'
; CHECK: LV: Not vectorizing: Loop doesn't have a legal pre-header.
entry:
  indirectbr ptr blockaddress(@outer_loop_preheader_indirectbr, %outer.header), [label %outer.header]

outer.header:
  %outer.iv = phi i64 [ 0, %entry ], [ %outer.iv.next, %outer.latch ]
  br label %inner

inner:
  %inner.iv = phi i64 [ 0, %outer.header ], [ %inner.iv.next, %inner ]
  %inner.iv.next = add nuw nsw i64 %inner.iv, 1
  %inner.ec = icmp eq i64 %inner.iv.next, 8
  br i1 %inner.ec, label %outer.latch, label %inner

outer.latch:
  %outer.iv.next = add nuw nsw i64 %outer.iv, 1
  %outer.ec = icmp eq i64 %outer.iv.next, %n
  br i1 %outer.ec, label %exit, label %outer.header, !llvm.loop !0

exit:
  ret void
}

; Operand bundles are not modeled, so calls with them must be rejected.
define void @outer_loop_call_operand_bundle(ptr %p, i64 %n) {
; CHECK-LABEL: LV: Checking a loop in 'outer_loop_call_operand_bundle'
; CHECK: LV: Not vectorizing: Found a call with operand bundles
entry:
  br label %outer.header

outer.header:
  %outer.iv = phi i64 [ 0, %entry ], [ %outer.iv.next, %outer.latch ]
  %gep = getelementptr float, ptr %p, i64 %outer.iv
  br label %inner

inner:
  %inner.iv = phi i64 [ 0, %outer.header ], [ %inner.iv.next, %inner ]
  %l = load float, ptr %gep
  %s = call float @llvm.sin.f32(float %l) [ "deopt"(i64 %outer.iv) ]
  store float %s, ptr %gep
  %inner.iv.next = add nuw nsw i64 %inner.iv, 1
  %inner.ec = icmp eq i64 %inner.iv.next, 8
  br i1 %inner.ec, label %outer.latch, label %inner

outer.latch:
  %outer.iv.next = add nuw nsw i64 %outer.iv, 1
  %outer.ec = icmp eq i64 %outer.iv.next, %n
  br i1 %outer.ec, label %exit, label %outer.header, !llvm.loop !0

exit:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.vectorize.enable"}
