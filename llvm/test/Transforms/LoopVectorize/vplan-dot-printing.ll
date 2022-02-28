; REQUIRES: asserts

; RUN: opt -loop-vectorize -debug-only=loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -vplan-print-in-dot-format -disable-output %s 2>&1 | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; Verify that -vplan-print-in-dot-format option works.

define void @print_call_and_memory(i64 %n, float* noalias %y, float* noalias %x) nounwind uwtable {
; CHECK:       subgraph cluster_N0 {
; CHECK-NEXT:    fontname=Courier
; CHECK-NEXT:    label="\<x1\> vector loop"
; CHECK-NEXT:    N1 [label =
; CHECK-NEXT:    "for.body:\l" +
; CHECK-NEXT:    "  EMIT vp\<[[CAN_IV:%.+]]\> = CANONICAL-INDUCTION\l" +
; CHECK-NEXT:    "  ir\<%iv\> = SCALAR-STEPS vp\<[[CAN_IV]]\>, ir\<0\>, ir\<1\>\l" +
; CHECK-NEXT:    "  CLONE ir\<%arrayidx\> = getelementptr ir\<%y\>, ir\<%iv\>\l" +
; CHECK-NEXT:    "  WIDEN ir\<%lv\> = load ir\<%arrayidx\>\l" +
; CHECK-NEXT:    "  WIDEN-CALL ir\<%call\> = call @llvm.sqrt.f32(ir\<%lv\>)\l" +
; CHECK-NEXT:    "  CLONE ir\<%arrayidx2\> = getelementptr ir\<%x\>, ir\<%iv\>\l" +
; CHECK-NEXT:    "  WIDEN store ir\<%arrayidx2\>, ir\<%call\>\l" +
; CHECK-NEXT:    "  EMIT vp\<[[CAN_IV_NEXT:%.+]]\> = VF * UF +(nuw) vp\<[[CAN_IV]]\>\l" +
; CHECK-NEXT:    "  EMIT branch-on-count vp\<[[CAN_IV_NEXT]]\> vp\<{{.+}}\>\l" +
; CHECK-NEXT:    "No successors\l"
; CHECK-NEXT:  ]
;
entry:
  %cmp6 = icmp sgt i64 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, float* %y, i64 %iv
  %lv = load float, float* %arrayidx, align 4
  %call = tail call float @llvm.sqrt.f32(float %lv) nounwind readnone
  %arrayidx2 = getelementptr inbounds float, float* %x, i64 %iv
  store float %call, float* %arrayidx2, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare float @llvm.sqrt.f32(float) nounwind readnone
