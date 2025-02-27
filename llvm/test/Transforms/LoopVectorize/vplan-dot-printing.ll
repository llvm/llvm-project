; REQUIRES: asserts

; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -vplan-print-in-dot-format -disable-output %s 2>&1 | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; Verify that -vplan-print-in-dot-format option works.

define void @print_call_and_memory(i64 %n, ptr noalias %y, ptr noalias %x) nounwind uwtable {
; CHECK:      digraph VPlan {
; CHECK-NEXT:  graph [labelloc=t, fontsize=30; label="Vectorization Plan\nInitial VPlan for VF=\{4\},UF\>=1\nLive-in vp\<[[VFxUF:%.+]]\> = VF * UF\nLive-in vp\<[[VEC_TC:%.+]]\> = vector-trip-count\nLive-in ir\<%n\> = original trip-count\n"]
; CHECK-NEXT:  node [shape=rect, fontname=Courier, fontsize=30]
; CHECK-NEXT:  edge [fontname=Courier, fontsize=30]
; CHECK-NEXT:  compound=true
; CHECK-NEXT:  N0 [label =
; CHECK-NEXT:    "ir-bb\<for.body.preheader\>:\l" +
; CHECK-NEXT:    "Successor(s): vector.ph\l"
; CHECK-NEXT:  ]
; CHECK-NEXT:  N0 -> N1 [ label=""]
; CHECK-NEXT:  N1 [label =
; CHECK-NEXT:    "vector.ph:\l" +
; CHECK-NEXT:    "Successor(s): vector loop\l"
; CHECK-NEXT:  ]
; CHECK-NEXT:  N1 -> N2 [ label="" lhead=cluster_N3]
; CHECK-NEXT:  subgraph cluster_N3 {
; CHECK-NEXT:    fontname=Courier
; CHECK-NEXT:    label="\<x1\> vector loop"
; CHECK-NEXT:    N2 [label =
; CHECK-NEXT:    "vector.body:\l" +
; CHECK-NEXT:    "  EMIT vp\<[[CAN_IV:%.+]]\> = CANONICAL-INDUCTION ir\<0\>, vp\<[[CAN_IV_NEXT:%.+]]\>\l" +
; CHECK-NEXT:    "  vp\<[[STEPS:%.+]]\> = SCALAR-STEPS vp\<[[CAN_IV]]\>, ir\<1\>\l" +
; CHECK-NEXT:    "  CLONE ir\<%arrayidx\> = getelementptr inbounds ir\<%y\>, vp\<[[STEPS]]\>\l" +
; CHECK-NEXT:    "  vp\<[[VEC_PTR:%.+]]\> = vector-pointer ir\<%arrayidx\>\l" +
; CHECK-NEXT:    "  WIDEN ir\<%lv\> = load vp\<[[VEC_PTR]]\>\l" +
; CHECK-NEXT:    "  WIDEN-INTRINSIC ir\<%call\> = call llvm.sqrt(ir\<%lv\>)\l" +
; CHECK-NEXT:    "  CLONE ir\<%arrayidx2\> = getelementptr inbounds ir\<%x\>, vp\<[[STEPS]]\>\l" +
; CHECK-NEXT:    "  vp\<[[VEC_PTR2:%.+]]\> = vector-pointer ir\<%arrayidx2\>\l" +
; CHECK-NEXT:    "  WIDEN store vp\<[[VEC_PTR2]]\>, ir\<%call\>\l" +
; CHECK-NEXT:    "  EMIT vp\<[[CAN_IV_NEXT]]\> = add nuw vp\<[[CAN_IV]]\>, vp\<[[VFxUF]]\>\l" +
; CHECK-NEXT:    "  EMIT branch-on-count vp\<[[CAN_IV_NEXT]]\>, vp\<[[VEC_TC]]\>\l" +
; CHECK-NEXT:    "No successors\l"
; CHECK-NEXT:  ]
;
entry:
  %cmp6 = icmp sgt i64 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, ptr %y, i64 %iv
  %lv = load float, ptr %arrayidx, align 4
  %call = tail call float @llvm.sqrt.f32(float %lv) nounwind readnone
  %arrayidx2 = getelementptr inbounds float, ptr %x, i64 %iv
  store float %call, ptr %arrayidx2, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare float @llvm.sqrt.f32(float) nounwind readnone
