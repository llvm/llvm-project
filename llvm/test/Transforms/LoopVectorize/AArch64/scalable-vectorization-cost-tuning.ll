; REQUIRES: asserts
; RUN: opt -mtriple=aarch64 -mattr=+sve \
; RUN:     -force-target-instruction-cost=1 -passes=loop-vectorize -S -debug-only=loop-vectorize < %s 2>&1 \
; RUN:     | FileCheck %s --check-prefixes=GENERIC,VF-VSCALE16

; RUN: opt -mtriple=aarch64 -mattr=+sve -mcpu=generic \
; RUN:     -force-target-instruction-cost=1 -passes=loop-vectorize -S -debug-only=loop-vectorize < %s 2>&1 \
; RUN:     | FileCheck %s --check-prefixes=GENERIC,VF-VSCALE16

; RUN: opt -mtriple=aarch64 -mcpu=neoverse-v1 \
; RUN:     -force-target-instruction-cost=1 -passes=loop-vectorize -S -debug-only=loop-vectorize < %s 2>&1 \
; RUN:     | FileCheck %s --check-prefixes=NEOVERSE-V1,VF-VSCALE16

; RUN: opt -mtriple=aarch64 -mcpu=neoverse-n2 \
; RUN:     -force-target-instruction-cost=1 -passes=loop-vectorize -S -debug-only=loop-vectorize < %s 2>&1 \
; RUN:     | FileCheck %s --check-prefixes=NEOVERSE-N2,VF-VSCALE16

; RUN: opt -mtriple=aarch64 -mcpu=neoverse-v2 \
; RUN:     -force-target-instruction-cost=1 -passes=loop-vectorize -S -debug-only=loop-vectorize < %s 2>&1 \
; RUN:     | FileCheck %s --check-prefixes=NEOVERSE-V2,VF-16

; GENERIC: Cost for VF vscale x 2: 11 (Estimated cost per lane: 2.
; GENERIC: Cost for VF vscale x 4: 11 (Estimated cost per lane: 1.
; GENERIC: LV: Selecting VF: vscale x 16

; NEOVERSE-V1: Cost for VF vscale x 2: 11 (Estimated cost per lane: 2.
; NEOVERSE-V1: Cost for VF vscale x 4: 11 (Estimated cost per lane: 1.
; NEOVERSE-V1: LV: Selecting VF: vscale x 16

; NEOVERSE-N2: Cost for VF vscale x 2: 11 (Estimated cost per lane: 5.
; NEOVERSE-N2: Cost for VF vscale x 4: 11 (Estimated cost per lane: 2.
; NEOVERSE-N2: LV: Selecting VF: vscale x 16

; NEOVERSE-V2: Cost for VF vscale x 2: 11 (Estimated cost per lane: 5.
; NEOVERSE-V2: Cost for VF vscale x 4: 11 (Estimated cost per lane: 2.
; NEOVERSE-V2: LV: Selecting VF: 16

; VF-16: <16 x i8>
; VF-VSCALE16: <vscale x 16 x i8>
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
