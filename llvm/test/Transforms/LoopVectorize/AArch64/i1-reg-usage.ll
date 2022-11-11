; RUN: opt -loop-vectorize -debug-only=loop-vectorize -disable-output -prefer-predicate-over-epilogue=scalar-epilogue 2>&1 < %s | FileCheck %s
; REQUIRES: asserts

target triple = "aarch64"

; Test that shows how many registers the loop vectorizer thinks an illegal <VF x i1> will consume.

; CHECK-LABEL: LV: Checking a loop in 'or_reduction_neon' from <stdin>
; CHECK: LV(REG): VF = 32
; CHECK-NEXT: LV(REG): Found max usage: 2 item
; CHECK-NEXT: LV(REG): RegisterClass: Generic::VectorRC, 72 registers
; CHECK-NEXT: LV(REG): RegisterClass: Generic::ScalarRC, 1 registers

define i1 @or_reduction_neon(i32 %arg, ptr %ptr) {
entry:
  br label %loop
exit:
  ret i1 %reduction_next
loop:
  %induction = phi i32 [ 0, %entry ], [ %induction_next, %loop ]
  %reduction = phi i1 [ 0, %entry ], [ %reduction_next, %loop ]
  %gep = getelementptr inbounds i32, ptr %ptr, i32 %induction
  %loaded = load i32, ptr %gep
  %i1 = icmp eq i32 %loaded, %induction
  %reduction_next = or i1 %i1, %reduction
  %induction_next = add nuw i32 %induction, 1
  %cond = icmp eq i32 %induction_next, %arg
  br i1 %cond, label %exit, label %loop, !llvm.loop !32
}

; CHECK-LABEL: LV: Checking a loop in 'or_reduction_sve'
; CHECK: LV(REG): VF = 64
; CHECK-NEXT: LV(REG): Found max usage: 2 item
; CHECK-NEXT: LV(REG): RegisterClass: Generic::VectorRC, 136 registers
; CHECK-NEXT: LV(REG): RegisterClass: Generic::ScalarRC, 1 registers

define i1 @or_reduction_sve(i32 %arg, ptr %ptr) vscale_range(2,2) "target-features"="+sve" {
entry:
  br label %loop
exit:
  ret i1 %reduction_next
loop:
  %induction = phi i32 [ 0, %entry ], [ %induction_next, %loop ]
  %reduction = phi i1 [ true, %entry ], [ %reduction_next, %loop ]
  %gep = getelementptr inbounds i32, ptr %ptr, i32 %induction
  %loaded = load i32, ptr %gep
  %i1 = icmp eq i32 %loaded, %induction
  %reduction_next = or i1 %i1, %reduction
  %induction_next = add nuw i32 %induction, 1
  %cond = icmp eq i32 %induction_next, %arg
  br i1 %cond, label %exit, label %loop, !llvm.loop !64
}

!32 = distinct !{!32, !33}
!33 = !{!"llvm.loop.vectorize.width", i32 32}
!64 = distinct !{!64, !65}
!65 = !{!"llvm.loop.vectorize.width", i32 64}
