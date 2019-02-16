; RUN: opt -mtriple=amdgcn-- -analyze -divergence -use-gpu-divergence-analysis %s | FileCheck %s

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
; CHECK-LABEL: Printing analysis 'Legacy Divergence Analysis' for function 'unstructured_loop'
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
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

declare i32 @llvm.amdgcn.workitem.id.x() #0

attributes #0 = { nounwind readnone }
