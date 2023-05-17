; RUN: opt < %s -passes=lowerswitch -S | FileCheck %s

; This test verifies -lowerswitch does not crash when an removing an
; unreachable default branch causes a PHI node used as the switch
; condition to be erased.

define void @f() local_unnamed_addr {
entry:
  br label %sw.epilog

sw.epilog:                                        ; preds = %sw.epilog.outer, %for.body
  %i = phi i32 [ undef, %for.body ], [ 0, %entry ]
  br i1 undef, label %for.body, label %for.end

for.body:                                         ; preds = %sw.epilog
  switch i32 %i, label %sw.epilog [
    i32 0, label %sw.epilog.outer.backedge.loopexit
    i32 1, label %sw.epilog.outer.backedge
  ]

sw.epilog.outer.backedge.loopexit:                ; preds = %for.body
  br label %for.end

sw.epilog.outer.backedge:                         ; preds = %for.body
  unreachable

for.end:                                          ; preds = %sw.epilog
  ret void
}

; The phi and the switch should both be eliminated.
; CHECK: @f()
; CHECK: sw.epilog:
; CHECK-NOT: phi
; CHECK: for.body:
; CHECK-NOT: switch
