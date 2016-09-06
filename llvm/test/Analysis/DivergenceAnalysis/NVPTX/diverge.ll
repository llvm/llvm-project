; RUN: opt %s -analyze -divergence | FileCheck %s

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; return (n < 0 ? a + threadIdx.x : b + threadIdx.x)
define i32 @no_diverge(i32 %n, i32 %a, i32 %b) {
; CHECK-LABEL: Printing analysis 'Divergence Analysis' for function 'no_diverge'
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %cond = icmp slt i32 %n, 0
  br i1 %cond, label %then, label %else ; uniform
; CHECK-NOT: DIVERGENT: br i1 %cond,
then:
  %a1 = add i32 %a, %tid
  br label %merge
else:
  %b2 = add i32 %b, %tid
  br label %merge
merge:
  %c = phi i32 [ %a1, %then ], [ %b2, %else ]
  ret i32 %c
}

; c = a;
; if (threadIdx.x < 5)    // divergent: data dependent
;   c = b;
; return c;               // c is divergent: sync dependent
define i32 @sync(i32 %a, i32 %b) {
; CHECK-LABEL: Printing analysis 'Divergence Analysis' for function 'sync'
bb1:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %cond = icmp slt i32 %tid, 5
  br i1 %cond, label %bb2, label %bb3
; CHECK: DIVERGENT: br i1 %cond,
bb2:
  br label %bb3
bb3:
  %c = phi i32 [ %a, %bb1 ], [ %b, %bb2 ] ; sync dependent on tid
; CHECK: DIVERGENT: %c =
  ret i32 %c
}

; c = 0;
; if (threadIdx.x >= 5) {  // divergent
;   c = (n < 0 ? a : b);  // c here is uniform because n is uniform
; }
; // c here is divergent because it is sync dependent on threadIdx.x >= 5
; return c;
define i32 @mixed(i32 %n, i32 %a, i32 %b) {
; CHECK-LABEL: Printing analysis 'Divergence Analysis' for function 'mixed'
bb1:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.z()
  %cond = icmp slt i32 %tid, 5
  br i1 %cond, label %bb6, label %bb2
; CHECK: DIVERGENT: br i1 %cond,
bb2:
  %cond2 = icmp slt i32 %n, 0
  br i1 %cond2, label %bb4, label %bb3
bb3:
  br label %bb5
bb4:
  br label %bb5
bb5:
  %c = phi i32 [ %a, %bb3 ], [ %b, %bb4 ]
; CHECK-NOT: DIVERGENT: %c =
  br label %bb6
bb6:
  %c2 = phi i32 [ 0, %bb1], [ %c, %bb5 ]
; CHECK: DIVERGENT: %c2 =
  ret i32 %c2
}

; We conservatively treats all parameters of a __device__ function as divergent.
define i32 @device(i32 %n, i32 %a, i32 %b) {
; CHECK-LABEL: Printing analysis 'Divergence Analysis' for function 'device'
; CHECK: DIVERGENT: i32 %n
; CHECK: DIVERGENT: i32 %a
; CHECK: DIVERGENT: i32 %b
entry:
  %cond = icmp slt i32 %n, 0
  br i1 %cond, label %then, label %else
; CHECK: DIVERGENT: br i1 %cond,
then:
  br label %merge
else:
  br label %merge
merge:
  %c = phi i32 [ %a, %then ], [ %b, %else ]
  ret i32 %c
}

; int i = 0;
; do {
;   i++;                  // i here is uniform
; } while (i < laneid);
; return i == 10 ? 0 : 1; // i here is divergent
;
; The i defined in the loop is used outside.
define i32 @loop() {
; CHECK-LABEL: Printing analysis 'Divergence Analysis' for function 'loop'
entry:
  %laneid = call i32 @llvm.nvvm.read.ptx.sreg.laneid()
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i1, %loop ]
; CHECK-NOT: DIVERGENT: %i =
  %i1 = add i32 %i, 1
  %exit_cond = icmp sge i32 %i1, %laneid
  br i1 %exit_cond, label %loop_exit, label %loop
loop_exit:
  %cond = icmp eq i32 %i, 10
  br i1 %cond, label %then, label %else
; CHECK: DIVERGENT: br i1 %cond,
then:
  ret i32 0
else:
  ret i32 1
}

; Same as @loop, but the loop is in the LCSSA form.
define i32 @lcssa() {
; CHECK-LABEL: Printing analysis 'Divergence Analysis' for function 'lcssa'
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i1, %loop ]
; CHECK-NOT: DIVERGENT: %i =
  %i1 = add i32 %i, 1
  %exit_cond = icmp sge i32 %i1, %tid
  br i1 %exit_cond, label %loop_exit, label %loop
loop_exit:
  %i.lcssa = phi i32 [ %i, %loop ]
; CHECK: DIVERGENT: %i.lcssa =
  %cond = icmp eq i32 %i.lcssa, 10
  br i1 %cond, label %then, label %else
; CHECK: DIVERGENT: br i1 %cond,
then:
  ret i32 0
else:
  ret i32 1
}

; This test contains an unstructured loop.
;           +-------------- entry ----------------+
;           |                                     |
;           V                                     V
; i1 = phi(0, i3)                            i2 = phi(0, i3)
;     j1 = i1 + 1 ---> i3 = phi(j1, j2) <--- j2 = i2 + 2
;           ^                 |                   ^
;           |                 V                   |
;           +-------- switch (tid / i3) ----------+
;                             |
;                             V
;                        if (i3 == 5) // divergent
; because sync dependent on (tid / i3).
define i32 @unstructured_loop(i1 %entry_cond) {
; CHECK-LABEL: Printing analysis 'Divergence Analysis' for function 'unstructured_loop'
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  br i1 %entry_cond, label %loop_entry_1, label %loop_entry_2
loop_entry_1:
  %i1 = phi i32 [ 0, %entry ], [ %i3, %loop_latch ]
  %j1 = add i32 %i1, 1
  br label %loop_body
loop_entry_2:
  %i2 = phi i32 [ 0, %entry ], [ %i3, %loop_latch ]
  %j2 = add i32 %i2, 2
  br label %loop_body
loop_body:
  %i3 = phi i32 [ %j1, %loop_entry_1 ], [ %j2, %loop_entry_2 ]
  br label %loop_latch
loop_latch:
  %div = sdiv i32 %tid, %i3
  switch i32 %div, label %branch [ i32 1, label %loop_entry_1
                                   i32 2, label %loop_entry_2 ]
branch:
  %cmp = icmp eq i32 %i3, 5
  br i1 %cmp, label %then, label %else
; CHECK: DIVERGENT: br i1 %cmp,
then:
  ret i32 0
else:
  ret i32 1
}

; Verifies sync-dependence is computed correctly in the absense of loops.
define i32 @sync_no_loop(i32 %arg) {
entry:
  %0 = add i32 %arg, 1
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %1 = icmp sge i32 %tid, 10
  br i1 %1, label %bb1, label %bb2

bb1:
  br label %bb3

bb2:
  br label %bb3

bb3:
  %2 = add i32 %0, 2
  ; CHECK-NOT: DIVERGENT: %2
  ret i32 %2
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.z()
declare i32 @llvm.nvvm.read.ptx.sreg.laneid()

!nvvm.annotations = !{!0, !1, !2, !3, !4, !5}
!0 = !{i32 (i32, i32, i32)* @no_diverge, !"kernel", i32 1}
!1 = !{i32 (i32, i32)* @sync, !"kernel", i32 1}
!2 = !{i32 (i32, i32, i32)* @mixed, !"kernel", i32 1}
!3 = !{i32 ()* @loop, !"kernel", i32 1}
!4 = !{i32 (i1)* @unstructured_loop, !"kernel", i32 1}
!5 = !{i32 (i32)* @sync_no_loop, !"kernel", i32 1}
