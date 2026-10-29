; RUN: opt -S -passes=sink < %s | FileCheck %s --check-prefixes=CHECK,DEFAULT
; RUN: opt -S -passes='sink<avoid-postdominators>' < %s | FileCheck %s --check-prefixes=CHECK,AVOID

; %m is only used after the diamond. Sinking it into the merge block saves no
; execution and keeps both operands live across the diamond.
define float @no_sink_into_postdominator(float %x, float %y, i1 %c, ptr %p) {
; CHECK-LABEL: @no_sink_into_postdominator(
; CHECK-NEXT:  entry:
; AVOID-NEXT:    %m = fmul float %x, %y
; CHECK-NEXT:    br i1 %c, label %then, label %merge
; CHECK:       merge:
; DEFAULT-NEXT:  %m = fmul float %x, %y
; CHECK-NEXT:    ret float %m
;
entry:
  %m = fmul float %x, %y
  br i1 %c, label %then, label %merge

then:
  store float 0.0, ptr %p
  br label %merge

merge:
  ret float %m
}

; Both operands are live into the merge block anyway, so sinking %m there
; cannot extend any live range.
define float @sink_into_postdominator_operands_live(float %x, float %y, i1 %c,
                                                    ptr %p) {
; CHECK-LABEL: @sink_into_postdominator_operands_live(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 %c, label %then, label %merge
; CHECK:       merge:
; CHECK-NEXT:    %m = fmul float %x, %y
; CHECK-NEXT:    %a = fadd float %m, %x
; CHECK-NEXT:    %b = fadd float %a, %y
; CHECK-NEXT:    ret float %b
;
entry:
  %m = fmul float %x, %y
  br i1 %c, label %then, label %merge

then:
  store float 0.0, ptr %p
  br label %merge

merge:
  %a = fadd float %m, %x
  %b = fadd float %a, %y
  ret float %b
}

; The final destination is a conditional block, so sinking still pays off even
; though the first step goes through a block that postdominates the entry.
define float @sink_into_conditional_block(float %x, float %y, i1 %c) {
; CHECK-LABEL: @sink_into_conditional_block(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %if
; CHECK:       then:
; CHECK-NEXT:    %m = fmul float %x, %y
; CHECK-NEXT:    ret float %m
;
entry:
  %m = fmul float %x, %y
  br label %if

if:
  br i1 %c, label %then, label %else

then:
  ret float %m

else:
  ret float 0.0
}

; The nearest common dominator of the uses is in a loop, so Sink walks up to
; %a, which postdominates the entry. The move into %a must be checked too.
define void @no_sink_into_postdominating_loop_ancestor(float %x, float %y,
                                                       i1 %c, ptr %p) {
; CHECK-LABEL: @no_sink_into_postdominating_loop_ancestor(
; CHECK-NEXT:  entry:
; AVOID-NEXT:    %m = fmul float %x, %y
; CHECK-NEXT:    br label %a
; CHECK:       a:
; DEFAULT-NEXT:  %m = fmul float %x, %y
; CHECK-NEXT:    br i1 %c, label %loop, label %exit
;
entry:
  %m = fmul float %x, %y
  br label %a

a:
  br i1 %c, label %loop, label %exit

loop:
  store float %m, ptr %p
  br i1 %c, label %loop, label %exit

exit:
  ret void
}

; %x and %y are used in %join, which %merge does not dominate but reaches, so
; they are live into %merge anyway and sinking %m extends nothing.
define float @sink_operands_live_through_later_join(float %x, float %y, i1 %c,
                                                    i1 %d, ptr %p) {
; CHECK-LABEL: @sink_operands_live_through_later_join(
; CHECK:       outer:
; CHECK-NEXT:    br i1 %c, label %then, label %merge
; CHECK:       merge:
; CHECK-NEXT:    %m = fmul float %x, %y
; CHECK-NEXT:    store float %m, ptr %p
;
entry:
  br i1 %d, label %outer, label %join

outer:
  %m = fmul float %x, %y
  br i1 %c, label %then, label %merge

then:
  store float 0.0, ptr %p
  br label %merge

merge:
  store float %m, ptr %p
  br label %join

join:
  %s = fadd float %x, %y
  ret float %s
}

; %x and %y are only used by PHIs on the edge from %then, which %merge cannot
; reach, so they are not live into %merge.
define float @no_sink_phi_use_not_reachable(float %x, float %y, i1 %c) {
; CHECK-LABEL: @no_sink_phi_use_not_reachable(
; CHECK-NEXT:  entry:
; AVOID-NEXT:    %m = fmul float %x, %y
; CHECK-NEXT:    br i1 %c, label %then, label %merge
; CHECK:       merge:
; CHECK-NEXT:    %px = phi float [ %x, %then ], [ 0.000000e+00, %entry ]
; CHECK-NEXT:    %py = phi float [ %y, %then ], [ 0.000000e+00, %entry ]
; DEFAULT-NEXT:  %m = fmul float %x, %y
;
entry:
  %m = fmul float %x, %y
  br i1 %c, label %then, label %merge

then:
  br label %merge

merge:
  %px = phi float [ %x, %then ], [ 0.0, %entry ]
  %py = phi float [ %y, %then ], [ 0.0, %entry ]
  %a = fadd float %px, %py
  %b = fadd float %a, %m
  ret float %b
}

; %x and %y are used by a PHI on an edge out of %merge, so they are live into
; %merge.
define float @sink_phi_use_reachable(float %x, float %y, i1 %c, i1 %d) {
; CHECK-LABEL: @sink_phi_use_reachable(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 %c, label %then, label %merge
; CHECK:       merge:
; CHECK-NEXT:    %m = fmul float %x, %y
;
entry:
  %m = fmul float %x, %y
  br i1 %c, label %then, label %merge

then:
  br label %merge

merge:
  br i1 %d, label %left, label %join

left:
  br label %join

join:
  %px = phi float [ %x, %left ], [ %m, %merge ]
  %py = phi float [ %y, %left ], [ %m, %merge ]
  %a = fadd float %px, %py
  ret float %a
}

; In a loop, the uses of %x in the header come after its definition, so
; reaching them from %merge does not keep the old %x live.
define void @no_sink_use_after_redefinition(ptr %p, float %y, i1 %c) {
; CHECK-LABEL: @no_sink_use_after_redefinition(
; CHECK:       loop:
; CHECK-NEXT:    %x = load float, ptr %p
; CHECK-NEXT:    store float %x, ptr %p
; AVOID-NEXT:    %m = fmul float %x, %y
; CHECK-NEXT:    br i1 %c, label %then, label %merge
; CHECK:       merge:
; DEFAULT-NEXT:  %m = fmul float %x, %y
; CHECK-NEXT:    store volatile float %m, ptr %p
;
entry:
  br label %loop

loop:
  %x = load float, ptr %p
  store float %x, ptr %p
  %m = fmul float %x, %y
  br i1 %c, label %then, label %merge

then:
  store float 0.0, ptr %p
  br label %merge

merge:
  store volatile float %m, ptr %p
  br i1 %c, label %loop, label %exit

exit:
  ret void
}

; A load is only sunk into blocks with a single predecessor, so %v first moves
; into %node, which postdominates the entry, and then on to %use over later
; iterations. Since it ends up in a conditional block, sinking pays off.
define float @sink_load_through_postdominator(ptr %p, i1 %c, i1 %d) {
; CHECK-LABEL: @sink_load_through_postdominator(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %node
; CHECK:       use:
; CHECK-NEXT:    %v = load float, ptr %p
; CHECK-NEXT:    ret float %v
;
entry:
  %v = load float, ptr %p
  br label %node

node:
  br i1 %c, label %leaf, label %exit

leaf:
  br i1 %d, label %use, label %exit

use:
  ret float %v

exit:
  ret float 0.0
}

; As above, but the store in %node may clobber %p, so once %v is in %node it
; cannot move on to %use. Sinking would only move %v into a postdominator.
define float @no_sink_load_blocked_by_store_in_postdominator(ptr %p, ptr %q,
                                                             i1 %c, i1 %d) {
; CHECK-LABEL: @no_sink_load_blocked_by_store_in_postdominator(
; CHECK-NEXT:  entry:
; AVOID-NEXT:    %v = load float, ptr %p
; CHECK-NEXT:    br label %node
; CHECK:       node:
; DEFAULT-NEXT:  %v = load float, ptr %p
; CHECK-NEXT:    store float 0.000000e+00, ptr %q
;
entry:
  %v = load float, ptr %p
  br label %node

node:
  store float 0.0, ptr %q
  br i1 %c, label %leaf, label %exit

leaf:
  br i1 %d, label %use, label %exit

use:
  ret float %v

exit:
  ret float 0.0
}
