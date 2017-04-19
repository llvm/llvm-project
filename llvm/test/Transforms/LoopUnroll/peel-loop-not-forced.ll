; RUN: opt < %s -S -loop-unroll -unroll-threshold=30 | FileCheck %s

define i32 @invariant_backedge_1(i32 %a, i32 %b) {
; CHECK-LABEL: @invariant_backedge_1
; CHECK-NOT:     %plus = phi
; CHECK:       loop.peel:
; CHECK:       loop:
; CHECK:         %i = phi
; CHECK:         %sum = phi
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %inc, %loop ]
  %sum = phi i32 [ 0, %entry ], [ %incsum, %loop ]
  %plus = phi i32 [ %a, %entry ], [ %b, %loop ]

  %incsum = add i32 %sum, %plus
  %inc = add i32 %i, 1
  %cmp = icmp slt i32 %i, 1000

  br i1 %cmp, label %loop, label %exit

exit:
  ret i32 %sum
}

define i32 @invariant_backedge_2(i32 %a, i32 %b) {
; This loop should be peeled twice because it has a Phi which becomes invariant
; starting from 3rd iteration.
; CHECK-LABEL: @invariant_backedge_2
; CHECK:       loop.peel{{.*}}:
; CHECK:       loop.peel{{.*}}:
; CHECK:         %i = phi
; CHECK:         %sum = phi
; CHECK-NOT:     %half.inv = phi
; CHECK-NOT:     %plus = phi
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %inc, %loop ]
  %sum = phi i32 [ 0, %entry ], [ %incsum, %loop ]
  %half.inv = phi i32 [ %a, %entry ], [ %b, %loop ]
  %plus = phi i32 [ %a, %entry ], [ %half.inv, %loop ]

  %incsum = add i32 %sum, %plus
  %inc = add i32 %i, 1
  %cmp = icmp slt i32 %i, 1000

  br i1 %cmp, label %loop, label %exit

exit:
  ret i32 %sum
}

define i32 @invariant_backedge_3(i32 %a, i32 %b) {
; This loop should be peeled thrice because it has a Phi which becomes invariant
; starting from 4th iteration.
; CHECK-LABEL: @invariant_backedge_3
; CHECK:       loop.peel{{.*}}:
; CHECK:       loop.peel{{.*}}:
; CHECK:       loop.peel{{.*}}:
; CHECK:         %i = phi
; CHECK:         %sum = phi
; CHECK-NOT:     %half.inv = phi
; CHECK-NOT:     %half.inv.2 = phi
; CHECK-NOT:     %plus = phi
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %inc, %loop ]
  %sum = phi i32 [ 0, %entry ], [ %incsum, %loop ]
  %half.inv = phi i32 [ %a, %entry ], [ %b, %loop ]
  %half.inv.2 = phi i32 [ %a, %entry ], [ %half.inv, %loop ]
  %plus = phi i32 [ %a, %entry ], [ %half.inv.2, %loop ]

  %incsum = add i32 %sum, %plus
  %inc = add i32 %i, 1
  %cmp = icmp slt i32 %i, 1000

  br i1 %cmp, label %loop, label %exit

exit:
  ret i32 %sum
}

define i32 @invariant_backedge_limited_by_size(i32 %a, i32 %b) {
; This loop should normally be peeled thrice because it has a Phi which becomes
; invariant starting from 4th iteration, but the size of the loop only allows
; us to peel twice because we are restricted to 30 instructions in resulting
; code. Thus, %plus Phi node should stay in loop even despite its backedge
; input is an invariant.
; CHECK-LABEL: @invariant_backedge_limited_by_size
; CHECK:       loop.peel{{.*}}:
; CHECK:       loop.peel{{.*}}:
; CHECK:         %i = phi
; CHECK:         %sum = phi
; CHECK:         %plus = phi i32 [ %a, {{.*}} ], [ %b, %loop ]
; CHECK-NOT:     %half.inv = phi
; CHECK-NOT:     %half.inv.2 = phi
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %inc, %loop ]
  %sum = phi i32 [ 0, %entry ], [ %incsum, %loop ]
  %half.inv = phi i32 [ %a, %entry ], [ %b, %loop ]
  %half.inv.2 = phi i32 [ %a, %entry ], [ %half.inv, %loop ]
  %plus = phi i32 [ %a, %entry ], [ %half.inv.2, %loop ]

  %incsum = add i32 %sum, %plus
  %inc = add i32 %i, 1
  %cmp = icmp slt i32 %i, 1000

  %incsum2 = add i32 %incsum, %plus
  %incsum3 = add i32 %incsum, %plus
  %incsum4 = add i32 %incsum, %plus
  %incsum5 = add i32 %incsum, %plus
  %incsum6 = add i32 %incsum, %plus
  %incsum7 = add i32 %incsum, %plus

  br i1 %cmp, label %loop, label %exit

exit:
  ret i32 %sum
}

; Peeling should fail due to method size.
define i32 @invariant_backedge_negative(i32 %a, i32 %b) {
; CHECK-LABEL: @invariant_backedge_negative
; CHECK-NOT:   loop.peel{{.*}}:
; CHECK:       loop:
; CHECK:         %i = phi
; CHECK:         %sum = phi
; CHECK:         %plus = phi
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %inc, %loop ]
  %sum = phi i32 [ 0, %entry ], [ %incsum2, %loop ]
  %plus = phi i32 [ %a, %entry ], [ %b, %loop ]

  %incsum = add i32 %sum, %plus
  %incsum2 = add i32 %incsum, %plus
  %incsum3 = add i32 %incsum, %plus
  %incsum4 = add i32 %incsum, %plus
  %incsum5 = add i32 %incsum, %plus
  %incsum6 = add i32 %incsum, %plus
  %incsum7 = add i32 %incsum, %plus
  %incsum8 = add i32 %incsum, %plus
  %incsum9 = add i32 %incsum, %plus
  %incsum10 = add i32 %incsum, %plus
  %incsum11 = add i32 %incsum, %plus
  %incsum12 = add i32 %incsum, %plus
  %incsum13 = add i32 %incsum, %plus
  %incsum14 = add i32 %incsum, %plus
  %incsum15 = add i32 %incsum, %plus
  %inc = add i32 %i, 1
  %cmp = icmp slt i32 %i, 1000

  br i1 %cmp, label %loop, label %exit

exit:
  ret i32 %sum
}

define i32 @cycled_phis(i32 %a, i32 %b) {
; Make sure that we do not crash working with cycled Phis and don't peel it.
; TODO: Actually this loop should be partially unrolled with factor 2.
; CHECK-LABEL: @cycled_phis
; CHECK-NOT:   loop.peel{{.*}}:
; CHECK:       loop:
; CHECK:         %i = phi
; CHECK:         %phi.a = phi
; CHECK:         %phi.b = phi
; CHECK:         %sum = phi
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %inc, %loop ]
  %phi.a = phi i32 [ %a, %entry ], [ %phi.b, %loop ]
  %phi.b = phi i32 [ %b, %entry ], [ %phi.a, %loop ]
  %sum = phi i32 [ 0, %entry], [ %incsum, %loop ]
  %incsum = add i32 %sum, %phi.a
  %inc = add i32 %i, 1
  %cmp = icmp slt i32 %i, 1000

  br i1 %cmp, label %loop, label %exit

exit:
  ret i32 %sum
}
