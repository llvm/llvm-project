; REQUIRES: asserts
; RUN: opt -mtriple=aarch64 -mattr=+sve \
; RUN:     -force-target-instruction-cost=1 -passes=loop-vectorize -S -debug-only=loop-vectorize < %s 2>&1 \
; RUN:     | FileCheck %s --check-prefixes=GENERIC,VF-VSCALE4

; RUN: opt -mtriple=aarch64 -mattr=+sve -mcpu=generic \
; RUN:     -force-target-instruction-cost=1 -passes=loop-vectorize -S -debug-only=loop-vectorize < %s 2>&1 \
; RUN:     | FileCheck %s --check-prefixes=GENERIC,VF-VSCALE4

; RUN: opt -mtriple=aarch64 -mcpu=neoverse-v1 \
; RUN:     -force-target-instruction-cost=1 -passes=loop-vectorize -S -debug-only=loop-vectorize < %s 2>&1 \
; RUN:     | FileCheck %s --check-prefixes=NEOVERSE-V1,VF-VSCALE4

; RUN: opt -mtriple=aarch64 -mcpu=neoverse-n2 \
; RUN:     -force-target-instruction-cost=1 -passes=loop-vectorize -S -debug-only=loop-vectorize < %s 2>&1 \
; RUN:     | FileCheck %s --check-prefixes=NEOVERSE-N2,VF-VSCALE4

; RUN: opt -mtriple=aarch64 -mcpu=neoverse-n2 \
; RUN:     -force-target-instruction-cost=1 -passes=loop-vectorize -S -debug-only=loop-vectorize < %s 2>&1 \
; RUN:     | FileCheck %s --check-prefixes=NEOVERSE-N2,VF-VSCALE4

; GENERIC: LV: Vector loop of width vscale x 2 costs: 3 (assuming a minimum vscale of 2).
; GENERIC: LV: Vector loop of width vscale x 4 costs: 1 (assuming a minimum vscale of 2).

; NEOVERSE-V1: LV: Vector loop of width vscale x 2 costs: 3 (assuming a minimum vscale of 2).
; NEOVERSE-V1: LV: Vector loop of width vscale x 4 costs: 1 (assuming a minimum vscale of 2).

; NEOVERSE-N2: LV: Vector loop of width vscale x 2 costs: 6 (assuming a minimum vscale of 1).
; NEOVERSE-N2: LV: Vector loop of width vscale x 4 costs: 3 (assuming a minimum vscale of 1).

; VF-4: <4 x i32>
; VF-VSCALE4: <16 x i32>
define void @test0(ptr %a, ptr %b, ptr %c) #0 {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %arrayidx = getelementptr inbounds i32, ptr %c, i64 %iv
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i8, ptr %b, i64 %iv
  %1 = load i8, ptr %arrayidx2, align 4
  %zext = zext i8 %1 to i32
  %add = add nsw i32 %zext, %0
  %arrayidx5 = getelementptr inbounds i32, ptr %a, i64 %iv
  store i32 %add, ptr %arrayidx5, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 1024
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret void
}

