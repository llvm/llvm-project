; RUN: opt -S -passes=loop-rotate < %s -verify-loop-info | FileCheck %s
;
; Verify that LoopRotate preserves LoopSimplify form even in very peculiar loop
; structures. We manually validate the CFG with FileCheck because currently we
; can't cause a failure when LoopSimplify fails to be preserved.

define void @PR18643(i1 %arg) {
; CHECK-LABEL: @PR18643(
entry:
  br label %outer.header
; CHECK: br label %outer.header

outer.header:
; CHECK: outer.header:
  br i1 %arg, label %inner.header, label %outer.body
; CHECK-NEXT: br i1 {{[^,]*}}, label %[[INNER_PREROTATE_PREHEADER:[^,]*]], label %outer.body

; CHECK: [[INNER_PREROTATE_PREHEADER]]:
; CHECK-NEXT: br i1 {{[^,]*}}, label %[[INNER_PREROTATE_PREHEADER_SPLIT_RETURN:[^,]*]], label %[[INNER_ROTATED_PREHEADER:[^,]*]]

; CHECK: [[INNER_ROTATED_PREHEADER]]:
; CHECK-NEXT: br label %inner.body

inner.header:
; Now the latch!
; CHECK: inner.header:
  br i1 %arg, label %return, label %inner.body
; CHECK-NEXT: br i1 {{[^,]*}}, label %[[INNER_SPLIT_RETURN:[^,]*]], label %inner.body

inner.body:
; Now the header!
; CHECK: inner.body:
  br i1 %arg, label %outer.latch, label %inner.latch
; CHECK-NEXT: br i1 {{[^,]*}}, label %[[INNER_SPLIT_OUTER_LATCH:[^,]*]], label %inner.header

inner.latch:
; Dead!
  br label %inner.header

outer.body:
; CHECK: outer.body:
  br label %outer.latch
; CHECK-NEXT: br label %outer.latch

; L2 -> L1 exit edge needs a simplified exit block.
; CHECK: [[INNER_SPLIT_OUTER_LATCH]]:
; CHECK-NEXT: br label %outer.latch

outer.latch:
; CHECK: outer.latch:
  br label %outer.header
; CHECK-NEXT: br label %outer.header

; L1 -> L0 exit edge need sa simplified exit block.
; CHECK: [[INNER_PREROTATE_PREHEADER_SPLIT_RETURN]]:
; CHECK-NEXT: br label %return

; L2 -> L0 exit edge needs a simplified exit block.
; CHECK: [[INNER_SPLIT_RETURN]]:
; CHECK-NEXT: br label %return

return:
; CHECK: return:
  unreachable
}
