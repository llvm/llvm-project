; RUN: opt -passes=loop-vectorize -force-vector-interleave=1 -force-vector-width=1 \
; RUN:   -scalable-vectorization=on -vectorize-vector-loops -mtriple=aarch64 \
; RUN:   -mattr=+sve2p1 -S < %s | FileCheck %s


define void @vector_mul_invariant_vector_operand(ptr noalias %a, ptr noalias readonly %b, <8 x i16> %m) {
; CHECK-LABEL: define void @vector_mul_invariant_vector_operand(
; CHECK: vector.ph:
; CHECK: call <vscale x 8 x i16> @llvm.vector.broadcast.nxv8i16.v8i16(<8 x i16> %m)
; CHECK: vector.body:
; CHECK: mul <vscale x 8 x i16>
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %bp = getelementptr inbounds <8 x i16>, ptr %b, i64 %iv
  %v = load <8 x i16>, ptr %bp, align 16
  %r = mul <8 x i16> %m, %v
  %ap = getelementptr inbounds <8 x i16>, ptr %a, i64 %iv
  store <8 x i16> %r, ptr %ap, align 16
  %iv.next = add nuw nsw i64 %iv, 1
  %exit = icmp eq i64 %iv.next, 1024
  br i1 %exit, label %done, label %loop

done:
  ret void
}
