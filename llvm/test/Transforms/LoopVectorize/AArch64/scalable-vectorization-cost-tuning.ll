; REQUIRES: asserts
; RUN: opt -mtriple=aarch64 -mattr=+sve \
; RUN:     -force-target-instruction-cost=1 -passes=loop-vectorize -S -debug-only=loop-vectorize < %s 2>&1 \
; RUN:     | FileCheck %s --check-prefixes=VSCALEFORTUNING1

; RUN: opt -mtriple=aarch64 -mattr=+sve -mcpu=generic \
; RUN:     -force-target-instruction-cost=1 -passes=loop-vectorize -S -debug-only=loop-vectorize < %s 2>&1 \
; RUN:     | FileCheck %s --check-prefixes=VSCALEFORTUNING1

; RUN: opt -mtriple=aarch64 -mcpu=neoverse-v1 \
; RUN:     -force-target-instruction-cost=1 -passes=loop-vectorize -S -debug-only=loop-vectorize < %s 2>&1 \
; RUN:     | FileCheck %s --check-prefixes=VSCALEFORTUNING2

; RUN: opt -mtriple=aarch64 -mcpu=neoverse-n2 \
; RUN:     -force-target-instruction-cost=1 -passes=loop-vectorize -S -debug-only=loop-vectorize < %s 2>&1 \
; RUN:     | FileCheck %s --check-prefixes=VSCALEFORTUNING1

; RUN: opt -mtriple=aarch64 -mcpu=neoverse-v2 \
; RUN:     -force-target-instruction-cost=1 -passes=loop-vectorize -S -debug-only=loop-vectorize < %s 2>&1 \
; RUN:     | FileCheck %s --check-prefixes=NEOVERSEV2

; VSCALEFORTUNING1: Cost for VF vscale x 2: 11 (Estimated cost per lane: 5.
; VSCALEFORTUNING1: Cost for VF vscale x 4: 11 (Estimated cost per lane: 2.
; VSCALEFORTUNING1: LV: Selecting VF: vscale x 16

; VSCALEFORTUNING2: Cost for VF vscale x 2: 11 (Estimated cost per lane: 2.
; VSCALEFORTUNING2: Cost for VF vscale x 4: 11 (Estimated cost per lane: 1.
; VSCALEFORTUNING2: LV: Selecting VF: vscale x 16

; NEOVERSEV2: Cost for VF vscale x 2: 11 (Estimated cost per lane: 5.
; NEOVERSEV2: Cost for VF vscale x 4: 11 (Estimated cost per lane: 2.
; NEOVERSEV2: LV: Selecting VF: 16

; VSCALEFORTUNING1: <vscale x 16 x i8>
; VSCALEFORTUNING2: <vscale x 16 x i8>
; NEOVERSEV2: <16 x i8>
define void @test0(ptr %a, ptr %b, ptr %c) #0 {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %arrayidx = getelementptr inbounds i8, ptr %c, i64 %iv
  %0 = load i8, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i8, ptr %b, i64 %iv
  %1 = load i8, ptr %arrayidx2, align 4
  %add = add nsw i8 %0, %1
  %arrayidx5 = getelementptr inbounds i8, ptr %a, i64 %iv
  store i8 %add, ptr %arrayidx5, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 1024
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret void
}
