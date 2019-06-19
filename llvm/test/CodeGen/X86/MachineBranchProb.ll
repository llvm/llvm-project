; RUN: llc < %s -mtriple=x86_64-apple-darwin -print-after=finalize-isel -o /dev/null 2>&1 | FileCheck %s

;; Make sure a transformation in SelectionDAGBuilder that converts "or + br" to
;; two branches correctly updates the branch probability.

@max_regno = common global i32 0, align 4

define void @test(i32* %old, i32 %final) {
for.cond:
  br label %for.cond2

for.cond2:                                        ; preds = %for.inc, %for.cond
  %i.1 = phi i32 [ %inc19, %for.inc ], [ 0, %for.cond ]
  %bit.0 = phi i32 [ %shl, %for.inc ], [ 1, %for.cond ]
  %tobool = icmp eq i32 %bit.0, 0
  %v3 = load i32, i32* @max_regno, align 4
  %cmp4 = icmp eq i32 %i.1, %v3
  %or.cond = or i1 %tobool, %cmp4
  br i1 %or.cond, label %for.inc20, label %for.inc, !prof !0
; CHECK: bb.1.for.cond2:
; CHECK: successors: %bb.3(0x01f3d4c5), %bb.4(0x7e0c2b3b)
; CHECK: bb.4.for.cond2:
; CHECK: successors: %bb.3(0x01fb92cf), %bb.2(0x7e046d31)

for.inc:                                          ; preds = %for.cond2
  %shl = shl i32 %bit.0, 1
  %inc19 = add nsw i32 %i.1, 1
  br label %for.cond2

for.inc20:                                        ; preds = %for.cond2
  ret void
}

!0 = !{!"branch_weights", i32 112017436, i32 -735157296}
