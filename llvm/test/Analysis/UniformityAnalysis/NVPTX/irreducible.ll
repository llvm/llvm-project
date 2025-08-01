; RUN: opt %s -passes='print<uniformity>' -disable-output 2>&1 | FileCheck %s

; NOTE: The new pass manager does not fall back on legacy divergence
; analysis even when the function contains an irreducible loop. The
; (new) divergence analysis conservatively reports all values as
; divergent. This test does not check for this conservative
; behaviour. Instead, it only checks for the values that are known to
; be divergent according to the legacy analysis.

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

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
define ptx_kernel i32 @unstructured_loop(i1 %entry_cond) {
; CHECK-LABEL: for function 'unstructured_loop'
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
; CHECK:  DIVERGENT: %cmp =
; CHECK: DIVERGENT: br i1 %cmp,
then:
  ret i32 0
else:
  ret i32 1
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.z()
declare i32 @llvm.nvvm.read.ptx.sreg.laneid()

