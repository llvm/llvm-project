; RUN: llc -asm-verbose=false < %s | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv8.2a-arm-unknown-eabihf"

define dso_local void @vec8(half* nocapture readonly %V, i32 %N) local_unnamed_addr #0 {
; CHECK:      .LBB0_1:
; CHECK-NEXT: vld1.16 {d16, d17}, [r0]!
; CHECK-NEXT: subs r1, r1, #8
; CHECK-NEXT: bne .LBB0_1
entry:
  br label %vector.body

vector.body:
  %index = phi i32 [ 0, %entry ], [ %index.next, %vector.body ]
  %0 = getelementptr inbounds half, half* %V, i32 %index
  %1 = bitcast half* %0 to <8 x half>*
  %wide.load = load volatile <8 x half>, <8 x half>* %1, align 2
  %index.next = add i32 %index, 8
  %cmp = icmp eq i32 %index.next, %N
  br i1 %cmp, label %byeblock, label %vector.body

byeblock:
  ret void
}

define dso_local void @vec4(half* nocapture readonly %V, i32 %N) local_unnamed_addr #0 {
; CHECK:      .LBB1_1:
; CHECK-NEXT: vld1.16 {d16}, [r0]!
; CHECK-NEXT: subs r1, r1, #4
; CHECK-NEXT:	bne	.LBB1_1
entry:
  br label %vector.body

vector.body:
  %index = phi i32 [ 0, %entry ], [ %index.next, %vector.body ]
  %0 = getelementptr inbounds half, half* %V, i32 %index
  %1 = bitcast half* %0 to <4 x half>*
  %wide.load = load volatile <4 x half>, <4 x half>* %1, align 2
  %index.next = add i32 %index, 4
  %cmp = icmp eq i32 %index.next, %N
  br i1 %cmp, label %byeblock, label %vector.body

byeblock:
  ret void
}

attributes #0 = { norecurse nounwind readonly "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "target-cpu"="generic" "target-features"="+armv8.2-a,+fullfp16,+strict-align,-thumb-mode" "unsafe-fp-math"="true" "use-soft-float"="false" }
