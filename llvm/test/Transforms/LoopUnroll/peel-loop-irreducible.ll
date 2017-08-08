; RUN: opt < %s -S -loop-unroll -unroll-force-peel-count=1 | FileCheck %s

; Check we don't peel loops where the latch is not the exiting block.
; CHECK-LABEL: @invariant_backedge_irreducible
; CHECK: entry:
; CHECK: br label %header
; CHECK-NOT: peel
; CHECK: header:
; CHECK: br i1 {{.*}} label %latch, label %exiting
; CHECK: latch:
; CHECK: br i1 {{.*}} label %header, label %exiting
; CHECK: exiting:
; CHECK: br i1 {{.*}} label %latch, label %exit

define i32 @invariant_backedge_irreducible(i32 %a, i32 %b) {
entry:
  br label %header

header:
  %i = phi i32 [ 0, %entry ], [ %inc, %latch ]
  %cmp.phi = phi i1 [ false, %entry ], [ %cmp, %latch ]
  br i1 %cmp.phi, label %latch, label %exiting

latch:
  %inc = add i32 %i, 1
  %cmp = icmp slt i32 %i, 1000
  br i1 %cmp, label %header, label %exiting

exiting:
  %cmp.exiting = phi i1 [ %cmp.phi, %header ], [ %cmp, %latch ]
  br i1 %cmp.exiting, label %latch, label %exit

exit:
  ret i32 0
}

