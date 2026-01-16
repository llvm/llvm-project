; RUN: opt -passes=loop-vectorize -force-vector-interleave=1 -force-vector-width=1 \
; RUN:   -scalable-vectorization=on -vectorize-vector-loops -mtriple=aarch64 \
; RUN:   -mattr=+sve2p1 -S < %s | FileCheck %s

define void @nontemporal_vector_load_store(ptr noalias %a, ptr noalias readonly %b) {
; CHECK-LABEL: define void @nontemporal_vector_load_store(
; CHECK: vector.body:
; CHECK: load <vscale x 8 x i16>, ptr {{.*}}, align 16, !nontemporal
; CHECK: store <vscale x 8 x i16> {{.*}}, ptr {{.*}}, align 16, !nontemporal
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %bp = getelementptr inbounds <8 x i16>, ptr %b, i64 %iv
  %v = load <8 x i16>, ptr %bp, align 16, !nontemporal !0
  %r = add <8 x i16> %v, splat (i16 1)
  %ap = getelementptr inbounds <8 x i16>, ptr %a, i64 %iv
  store <8 x i16> %r, ptr %ap, align 16, !nontemporal !0
  %iv.next = add nuw nsw i64 %iv, 1
  %exit = icmp eq i64 %iv.next, 1024
  br i1 %exit, label %done, label %loop

done:
  ret void
}

!0 = !{i32 1}
