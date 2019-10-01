; RUN: opt -S -licm < %s | FileCheck %s -check-prefixes=CHECK,CHECK-DISABLED
; RUN: opt -S -licm -licm-control-flow-hoisting=1 < %s | FileCheck %s -check-prefixes=CHECK,CHECK-ENABLED
; RUN: opt -S -licm -licm-control-flow-hoisting=0 < %s | FileCheck %s -check-prefixes=CHECK,CHECK-DISABLED
; RUN: opt -passes='require<opt-remark-emit>,loop(licm)' -S < %s | FileCheck %s -check-prefixes=CHECK,CHECK-DISABLED
; RUN: opt -passes='require<opt-remark-emit>,loop(licm)' -licm-control-flow-hoisting=1 -S < %s | FileCheck %s -check-prefixes=CHECK,CHECK-ENABLED
; RUN: opt -passes='require<opt-remark-emit>,loop(licm)' -licm-control-flow-hoisting=0 -S < %s | FileCheck %s -check-prefixes=CHECK,CHECK-DISABLED

; RUN: opt -passes='require<opt-remark-emit>,loop(licm)' -licm-control-flow-hoisting=1 -enable-mssa-loop-dependency=true -verify-memoryssa -S < %s | FileCheck %s -check-prefixes=CHECK,CHECK-ENABLED
; Enable run below when adding promotion. e.g. "store i32 %phi, i32* %p" is promoted to phi.lcssa.
; opt -passes='require<opt-remark-emit>,loop(licm)' -licm-control-flow-hoisting=0 -enable-mssa-loop-dependency=true -verify-memoryssa -S < %s | FileCheck %s -check-prefixes=CHECK,CHECK-DISABLED


; CHECK-LABEL: @triangle_phi
define void @triangle_phi(i32 %x, i32* %p) {
; CHECK-LABEL: entry:
; CHECK: %cmp1 = icmp sgt i32 %x, 0
; CHECK-ENABLED: br i1 %cmp1, label %[[IF_LICM:.*]], label %[[THEN_LICM:.*]]
entry:
  br label %loop

; CHECK-ENABLED: [[IF_LICM]]:
; CHECK: %add = add i32 %x, 1
; CHECK-ENABLED: br label %[[THEN_LICM]]

; CHECK-ENABLED: [[THEN_LICM]]:
; CHECK-ENABLED: phi i32 [ %add, %[[IF_LICM]] ], [ %x, %entry ]
; CHECK-ENABLED: store i32 %phi, i32* %p
; CHECK-ENABLED: %cmp2 = icmp ne i32 %phi, 0
; CHECK: br label %loop

loop:
  %cmp1 = icmp sgt i32 %x, 0
  br i1 %cmp1, label %if, label %then

if:
  %add = add i32 %x, 1
  br label %then

; CHECK-LABEL: then:
; CHECK-DISABLED: %phi = phi i32 [ %add, %if ], [ %x, %loop ]
; CHECK-DISABLED: %cmp2 = icmp ne i32 %phi, 0
then:
  %phi = phi i32 [ %add, %if ], [ %x, %loop ]
  store i32 %phi, i32* %p
  %cmp2 = icmp ne i32 %phi, 0
  br i1 %cmp2, label %loop, label %end

; CHECK-LABEL: end:
; CHECK-DISABLED: %[[PHI_LCSSA:.*]] = phi i32 [ %phi, %then ]
; CHECK-DISABLED: store i32 %[[PHI_LCSSA]], i32* %p
end:
  ret void
}

; CHECK-LABEL: @diamond_phi
define void @diamond_phi(i32 %x, i32* %p) {
; CHECK-LABEL: entry:
; CHECK: %cmp1 = icmp sgt i32 %x, 0
; CHECK-ENABLED: br i1 %cmp1, label %[[IF_LICM:.*]], label %[[ELSE_LICM:.*]]
entry:
  br label %loop

; CHECK-ENABLED: [[IF_LICM]]:
; CHECK-DAG: %add = add i32 %x, 1
; CHECK-ENABLED: br label %[[THEN_LICM:.*]]

; CHECK-ENABLED: [[ELSE_LICM]]:
; CHECK-DAG: %sub = sub i32 %x, 1
; CHECK-ENABLED: br label %[[THEN_LICM]]

; CHECK-ENABLED: [[THEN_LICM]]
; CHECK-ENABLED: %phi = phi i32 [ %add, %[[IF_LICM]] ], [ %sub, %[[ELSE_LICM]] ]
; CHECK-ENABLED: store i32 %phi, i32* %p
; CHECK-ENABLED: %cmp2 = icmp ne i32 %phi, 0
; CHECK: br label %loop

loop:
  %cmp1 = icmp sgt i32 %x, 0
  br i1 %cmp1, label %if, label %else

if:
  %add = add i32 %x, 1
  br label %then

else:
  %sub = sub i32 %x, 1
  br label %then

; CHECK-LABEL: then:
; CHECK-DISABLED: %phi = phi i32 [ %add, %if ], [ %sub, %else ]
; CHECK-DISABLED: %cmp2 = icmp ne i32 %phi, 0
then:
  %phi = phi i32 [ %add, %if ], [ %sub, %else ]
  store i32 %phi, i32* %p
  %cmp2 = icmp ne i32 %phi, 0
  br i1 %cmp2, label %loop, label %end

; CHECK-LABEL: end:
; CHECK-DISABLED: %[[PHI_LCSSA:.*]] = phi i32 [ %phi, %then ]
; CHECK-DISABLED: store i32 %[[PHI_LCSSA]], i32* %p
end:
  ret void
}

; TODO: This is currently too complicated for us to be able to hoist the phi.
; CHECK-LABEL: @three_way_phi
define void @three_way_phi(i32 %x, i32* %p) {
; CHECK-LABEL: entry:
; CHECK-DAG: %cmp1 = icmp sgt i32 %x, 0
; CHECK-DAG: %add = add i32 %x, 1
; CHECK-DAG: %cmp2 = icmp sgt i32 %add, 0
; CHECK-ENABLED: br i1 %cmp1, label %[[IF_LICM:.*]], label %[[ELSE_LICM:.*]]

; CHECK-ENABLED: [[IF_LICM]]:
; CHECK-ENABLED: br label %[[THEN_LICM:.*]]

; CHECK-ENABLED: [[THEN_LICM]]:
; CHECK: %sub = sub i32 %x, 1
; CHECK: br label %loop

entry:
  br label %loop

loop:
  %cmp1 = icmp sgt i32 %x, 0
  br i1 %cmp1, label %if, label %then

if:
  %add = add i32 %x, 1
  %cmp2 = icmp sgt i32 %add, 0
  br i1 %cmp2, label %if.if, label %then

if.if:
  %sub = sub i32 %x, 1
  br label %then

then:
  %phi = phi i32 [ 0, %loop ], [ %add, %if ], [ %sub, %if.if ]
  store i32 %phi, i32* %p
  %cmp3 = icmp ne i32 %phi, 0
  br i1 %cmp3, label %loop, label %end

end:
  ret void
}

; TODO: This is currently too complicated for us to be able to hoist the phi.
; CHECK-LABEL: @tree_phi
define void @tree_phi(i32 %x, i32* %p) {
; CHECK-LABEL: entry:
; CHECK-DAG: %cmp1 = icmp sgt i32 %x, 0
; CHECK-DAG: %add = add i32 %x, 1
; CHECK-DAG: %cmp2 = icmp sgt i32 %add, 0
; CHECK-DAG: %sub = sub i32 %x, 1
; CHECK: br label %loop

entry:
  br label %loop

loop:
  %cmp1 = icmp sgt i32 %x, 0
  br i1 %cmp1, label %if, label %else

if:
  %add = add i32 %x, 1
  %cmp2 = icmp sgt i32 %add, 0
  br i1 %cmp2, label %if.if, label %if.else

if.if:
  br label %then

if.else:
  br label %then

else:
  %sub = sub i32 %x, 1
  br label %then

then:
  %phi = phi i32 [ %add, %if.if ], [ 0, %if.else ], [ %sub, %else ]
  store i32 %phi, i32* %p
  %cmp3 = icmp ne i32 %phi, 0
  br i1 %cmp3, label %loop, label %end

end:
  ret void
}

; TODO: We can hoist the first phi, but not the second.
; CHECK-LABEL: @phi_phi
define void @phi_phi(i32 %x, i32* %p) {
; CHECK-LABEL: entry:
; CHECK-DAG: %cmp1 = icmp sgt i32 %x, 0
; CHECK-DAG: %add = add i32 %x, 1
; CHECK-DAG: %cmp2 = icmp sgt i32 %add, 0
; CHECK-DAG: %sub = sub i32 %x, 1
; CHECK-ENABLED: br i1 %cmp2, label %[[IF_IF_LICM:.*]], label %[[IF_ELSE_LICM:.*]]

; CHECK-ENABLED: [[IF_IF_LICM]]:
; CHECK-ENABLED: br label %[[IF_THEN_LICM:.*]]

; CHECK-ENABLED: [[IF_ELSE_LICM]]:
; CHECK-ENABLED: br label %[[IF_THEN_LICM]]

; CHECK-ENABLED: [[IF_THEN_LICM]]:
; CHECK-ENABLED: %phi1 = phi i32 [ %add, %[[IF_IF_LICM]] ], [ 0, %[[IF_ELSE_LICM]] ]
; CHECK: br label %loop

entry:
  br label %loop

loop:
  %cmp1 = icmp sgt i32 %x, 0
  br i1 %cmp1, label %if, label %else

if:
  %add = add i32 %x, 1
  %cmp2 = icmp sgt i32 %add, 0
  br i1 %cmp2, label %if.if, label %if.else

if.if:
  br label %if.then

if.else:
  br label %if.then

; CHECK-LABEL: if.then:
; CHECK-DISABLED: %phi1 = phi i32 [ %add, %if.if ], [ 0, %if.else ]
if.then:
  %phi1 = phi i32 [ %add, %if.if ], [ 0, %if.else ]
  br label %then

else:
  %sub = sub i32 %x, 1
  br label %then

; CHECK-LABEL: then:
; CHECK: %phi2 = phi i32 [ %phi1, %if.then ], [ %sub, %else ]
then:
  %phi2 = phi i32 [ %phi1, %if.then ], [ %sub, %else ]
  store i32 %phi2, i32* %p
  %cmp3 = icmp ne i32 %phi2, 0
  br i1 %cmp3, label %loop, label %end

end:
  ret void
}

; Check that we correctly duplicate empty control flow.
; CHECK-LABEL: @empty_triangle_phi
define i8 @empty_triangle_phi(i32 %x, i32 %y) {
; CHECK-LABEL: entry:
; CHECK: %cmp1 = icmp eq i32 %x, 0
; CHECK-ENABLED: br i1 %cmp1, label %[[IF_LICM:.*]], label %[[THEN_LICM:.*]]
entry:
  br label %loop

; CHECK-ENABLED: [[IF_LICM]]:
; CHECK-ENABLED: br label %[[THEN_LICM]]

; CHECK-ENABLED: [[THEN_LICM]]:
; CHECK-ENABLED: %phi = phi i8 [ 0, %[[IF_LICM]] ], [ 1, %entry ]
; CHECK: %cmp2 = icmp eq i32 %y, 0
; CHECK: br label %loop

loop:
  %cmp1 = icmp eq i32 %x, 0
  br i1 %cmp1, label %if, label %then

if:
  br label %then

; CHECK-LABEL: then:
; CHECK-DISABLED: %phi = phi i8 [ 0, %if ], [ 1, %loop ]
then:
  %phi = phi i8 [ 0, %if ], [ 1, %loop ]
  %cmp2 = icmp eq i32 %y, 0
  br i1 %cmp2, label %end, label %loop

end:
  ret i8 %phi
}

; CHECK-LABEL: @empty_diamond_phi
define i8 @empty_diamond_phi(i32 %x, i32 %y) {
; CHECK-LABEL: entry:
; CHECK: %cmp1 = icmp eq i32 %x, 0
; CHECK-ENABLED: br i1 %cmp1, label %[[IF_LICM:.*]], label %[[ELSE_LICM:.*]]
entry:
  br label %loop

; CHECK-ENABLED: [[IF_LICM]]:
; CHECK-ENABLED: br label %[[THEN_LICM:.*]]

; CHECK-ENABLED: [[ELSE_LICM]]:
; CHECK-ENABLED: br label %[[THEN_LICM]]

; CHECK-ENABLED: [[THEN_LICM]]:
; CHECK-ENABLED: %phi = phi i8 [ 0, %[[IF_LICM]] ], [ 1, %[[ELSE_LICM]] ]
; CHECK: %cmp2 = icmp eq i32 %y, 0
; CHECK: br label %loop

loop:
  %cmp1 = icmp eq i32 %x, 0
  br i1 %cmp1, label %if, label %else

if:
  br label %then

else:
  br label %then

; CHECK-LABEL: then:
; CHECK-DISABLED: %phi = phi i8 [ 0, %if ], [ 1, %else ]
then:
  %phi = phi i8 [ 0, %if ], [ 1, %else ]
  %cmp2 = icmp eq i32 %y, 0
  br i1 %cmp2, label %end, label %loop

end:
  ret i8 %phi
}

; Check that we correctly handle the case that the first thing we try to hoist is a phi.
; CHECK-LABEL: @empty_triangle_phi_first
define i8 @empty_triangle_phi_first(i32 %x, i1 %cond) {
; CHECK-LABEL: entry:
; CHECK-ENABLED: br i1 %cond, label %[[IF_LICM:.*]], label %[[THEN_LICM:.*]]
entry:
  br label %loop

; CHECK-ENABLED: [[IF_LICM]]:
; CHECK-ENABLED: br label %[[THEN_LICM]]

; CHECK-ENABLED: [[THEN_LICM]]:
; CHECK-ENABLED: %phi = phi i8 [ 0, %[[IF_LICM]] ], [ 1, %entry ]
; CHECK: %cmp = icmp eq i32 %x, 0
; CHECK: br label %loop

loop:
  br i1 %cond, label %if, label %then

if:
  br label %then

; CHECK-LABEL: then:
; CHECK-DISABLED: %phi = phi i8 [ 0, %if ], [ 1, %loop ]
then:
  %phi = phi i8 [ 0, %if ], [ 1, %loop ]
  %cmp = icmp eq i32 %x, 0
  br i1 %cmp, label %end, label %loop

end:
  ret i8 %phi
}

; CHECK-LABEL: @empty_diamond_phi
define i8 @empty_diamond_phi_first(i32 %x, i1 %cond) {
; CHECK-LABEL: entry:
; CHECK-ENABLED: br i1 %cond, label %[[IF_LICM:.*]], label %[[ELSE_LICM:.*]]
entry:
  br label %loop

; CHECK-ENABLED: [[IF_LICM]]:
; CHECK-ENABLED: br label %[[THEN_LICM:.*]]

; CHECK-ENABLED: [[ELSE_LICM]]:
; CHECK-ENABLED: br label %[[THEN_LICM]]

; CHECK-ENABLED: [[THEN_LICM]]:
; CHECK-ENABLED: %phi = phi i8 [ 0, %[[IF_LICM]] ], [ 1, %[[ELSE_LICM]] ]
; CHECK: %cmp = icmp eq i32 %x, 0
; CHECK: br label %loop

loop:
  br i1 %cond, label %if, label %else

if:
  br label %then

else:
  br label %then

; CHECK-LABEL: then:
; CHECK-DISABLED: %phi = phi i8 [ 0, %if ], [ 1, %else ]
then:
  %phi = phi i8 [ 0, %if ], [ 1, %else ]
  %cmp = icmp eq i32 %x, 0
  br i1 %cmp, label %end, label %loop

end:
  ret i8 %phi
}

; CHECK-LABEL: @empty_triangle_phi_first
define i8 @empty_triangle_phi_first_empty_loop_head(i32 %x, i1 %cond) {
; CHECK-LABEL: entry:
; CHECK-ENABLED: br i1 %cond, label %[[IF_LICM:.*]], label %[[THEN_LICM:.*]]
entry:
  br label %loop

; CHECK-ENABLED: [[IF_LICM]]:
; CHECK-ENABLED: br label %[[THEN_LICM]]

; CHECK-ENABLED: [[THEN_LICM]]:
; CHECK-ENABLED: %phi = phi i8 [ 0, %[[IF_LICM]] ], [ 1, %entry ]
; CHECK: %cmp = icmp eq i32 %x, 0
; CHECK: br label %loop

loop:
  br label %test

test:
  br i1 %cond, label %if, label %then

if:
  br label %then

; CHECK-LABEL: then:
; CHECK-DISABLED: %phi = phi i8 [ 0, %if ], [ 1, %test ]
then:
  %phi = phi i8 [ 0, %if ], [ 1, %test ]
  %cmp = icmp eq i32 %x, 0
  br i1 %cmp, label %end, label %loop

end:
  ret i8 %phi
}

; CHECK-LABEL: @empty_diamond_phi_first_empty_loop_head
define i8 @empty_diamond_phi_first_empty_loop_head(i32 %x, i1 %cond) {
; CHECK-LABEL: entry:
; CHECK-ENABLED: br i1 %cond, label %[[IF_LICM:.*]], label %[[ELSE_LICM:.*]]
entry:
  br label %loop

; CHECK-ENABLED: [[IF_LICM]]:
; CHECK-ENABLED: br label %[[THEN_LICM:.*]]

; CHECK-ENABLED: [[ELSE_LICM]]:
; CHECK-ENABLED: br label %[[THEN_LICM]]

; CHECK-ENABLED: [[THEN_LICM]]:
; CHECK-ENABLED: %phi = phi i8 [ 0, %[[IF_LICM]] ], [ 1, %[[ELSE_LICM]] ]
; CHECK: %cmp = icmp eq i32 %x, 0
; CHECK: br label %loop

loop:
  br label %test

test:
  br i1 %cond, label %if, label %else

if:
  br label %then

else:
  br label %then

; CHECK-LABEL: then:
; CHECK-DISABLED: %phi = phi i8 [ 0, %if ], [ 1, %else ]
then:
  %phi = phi i8 [ 0, %if ], [ 1, %else ]
  %cmp = icmp eq i32 %x, 0
  br i1 %cmp, label %end, label %loop

end:
  ret i8 %phi
}

; The phi is on one branch of a diamond while simultaneously at the end of a
; triangle. Check that we duplicate the triangle and not the diamond.
; CHECK-LABEL: @triangle_diamond
define void @triangle_diamond(i32* %ptr, i32 %x, i32 %y) {
; CHECK-LABEL: entry:
; CHECK-DAG: %cmp1 = icmp ne i32 %x, 0
; CHECK-DAG: %cmp2 = icmp ne i32 %y, 0
; CHECK-ENABLED: br i1 %cmp1, label %[[IF_LICM:.*]], label %[[THEN_LICM:.*]]
entry:
  br label %loop

; CHECK-ENABLED: [[IF_LICM]]:
; CHECK-ENABLED: br label %[[THEN_LICM]]

; CHECK-ENABLED: [[THEN_LICM]]:
; CHECK-ENABLED: %phi = phi i32 [ 0, %[[IF_LICM]] ], [ 127, %entry ]
; CHECK: br label %loop

loop:
  %cmp1 = icmp ne i32 %x, 0
  br i1 %cmp1, label %if, label %then

if:
  %cmp2 = icmp ne i32 %y, 0
  br i1 %cmp2, label %if.then, label %then

; CHECK-LABEL: then:
; CHECK-DISABLED: %phi = phi i32 [ 0, %if ], [ 127, %loop ]
then:
  %phi = phi i32 [ 0, %if ], [ 127, %loop ]
  store i32 %phi, i32* %ptr
  br label %end

if.then:
  br label %end

end:
  br label %loop
}

; As the previous, but the end of the diamond is the head of the loop.
; CHECK-LABEL: @triangle_diamond_backedge
define void @triangle_diamond_backedge(i32* %ptr, i32 %x, i32 %y) {
; CHECK-LABEL: entry:
; CHECK-DAG: %cmp1 = icmp ne i32 %x, 0
; CHECK-DAG: %cmp2 = icmp ne i32 %y, 0
; CHECK-ENABLED: br i1 %cmp1, label %[[IF_LICM:.*]], label %[[THEN_LICM:.*]]
entry:
  br label %loop

; CHECK-ENABLED: [[IF_LICM]]:
; CHECK-ENABLED: br label %[[THEN_LICM]]

; CHECK-ENABLED: [[THEN_LICM]]:
; CHECK-ENABLED: %phi = phi i32 [ 0, %[[IF_LICM]] ], [ 127, %entry ]
; CHECK: br label %loop

loop:
  %cmp1 = icmp ne i32 %x, 0
  br i1 %cmp1, label %if, label %then

if:
  %cmp2 = icmp ne i32 %y, 0
  br i1 %cmp2, label %backedge, label %then

; CHECK-LABEL: then:
; CHECK-DISABLED: %phi = phi i32 [ 0, %if ], [ 127, %loop ]
then:
  %phi = phi i32 [ 0, %if ], [ 127, %loop ]
  store i32 %phi, i32* %ptr
  br label %loop

backedge:
  br label %loop
}

; TODO: The inner diamonds can be hoisted, but not currently the outer diamond
; CHECK-LABEL: @diamonds_inside_diamond
define void @diamonds_inside_diamond(i32 %x, i32* %p) {
; CHECK-LABEL: entry:
; CHECK-DAG: %cmp1 = icmp sgt i32 %x, 0
; CHECK-DAG: %cmp3 = icmp slt i32 %x, -10
; CHECK-ENABLED: br i1 %cmp3, label %[[ELSE_IF_LICM:.*]], label %[[ELSE_ELSE_LICM:.*]]
entry:
  br label %loop

; CHECK-ENABLED: [[ELSE_IF_LICM]]:
; CHECK-ENABLED: br label %[[ELSE_THEN_LICM:.*]]

; CHECK-ENABLED: [[ELSE_ELSE_LICM]]:
; CHECK-ENABLED: br label %[[ELSE_THEN_LICM]]

; CHECK-ENABLED: [[ELSE_THEN_LICM]]:
; CHECK-ENABLED: %phi2 = phi i32 [ 2, %[[ELSE_IF_LICM]] ], [ 3, %[[ELSE_ELSE_LICM]] ]
; CHECK: %cmp2 = icmp sgt i32 %x, 10
; CHECK-ENABLED: br i1 %cmp2, label %[[IF_IF_LICM:.*]], label %[[IF_ELSE_LICM:.*]]

; CHECK-ENABLED: [[IF_IF_LICM]]:
; CHECK-ENABLED: br label %[[IF_THEN_LICM:.*]]

; CHECK-ENABLED: [[IF_ELSE_LICM]]:
; CHECK-ENABLED: br label %[[IF_THEN_LICM]]

; CHECK-ENABLED: [[IF_THEN_LICM]]:
; CHECK-ENABLED: %phi1 = phi i32 [ 0, %[[IF_IF_LICM]] ], [ 1, %[[IF_ELSE_LICM]] ]
; CHECK: br label %loop

loop:
  %cmp1 = icmp sgt i32 %x, 0
  br i1 %cmp1, label %if, label %else

if:
  %cmp2 = icmp sgt i32 %x, 10
  br i1 %cmp2, label %if.if, label %if.else

if.if:
  br label %if.then

if.else:
  br label %if.then

; CHECK-LABEL: if.then:
; CHECK-DISABLED: %phi1 = phi i32 [ 0, %if.if ], [ 1, %if.else ]
if.then:
  %phi1 = phi i32 [ 0, %if.if ], [ 1, %if.else ]
  br label %then

else:
  %cmp3 = icmp slt i32 %x, -10
  br i1 %cmp3, label %else.if, label %else.else

else.if:
  br label %else.then

else.else:
  br label %else.then

; CHECK-LABEL: else.then:
; CHECK-DISABLED: %phi2 = phi i32 [ 2, %else.if ], [ 3, %else.else ]
else.then:
  %phi2 = phi i32 [ 2, %else.if ], [ 3, %else.else ]
  br label %then

; CHECK-LABEL: then:
; CHECK: %phi3 = phi i32 [ %phi1, %if.then ], [ %phi2, %else.then ]
; CHECK: %cmp4 = icmp ne i32 %phi3, 0
then:
  %phi3 = phi i32 [ %phi1, %if.then ], [ %phi2, %else.then ]
  store i32 %phi3, i32* %p
  %cmp4 = icmp ne i32 %phi3, 0
  br i1 %cmp4, label %loop, label %end

end:
  ret void
}

; We can hoist blocks that contain an edge that exits the loop by ignoring that
; edge in the hoisted block.
; CHECK-LABEL: @triangle_phi_loopexit
define void @triangle_phi_loopexit(i32 %x, i32* %p) {
; CHECK-LABEL: entry:
; CHECK-DAG: %add = add i32 %x, 1
; CHECK-DAG: %cmp1 = icmp sgt i32 %x, 0
; CHECK-DAG: %cmp2 = icmp sgt i32 10, %add
; CHECK-ENABLED: br i1 %cmp1, label %[[IF_LICM:.*]], label %[[THEN_LICM:.*]]
entry:
  br label %loop

; CHECK-ENABLED: [[IF_LICM]]:
; CHECK-ENABLED: br label %[[THEN_LICM]]

; CHECK-ENABLED: [[THEN_LICM]]:
; CHECK-ENABLED: %phi = phi i32 [ %add, %[[IF_LICM]] ], [ %x, %entry ]
; CHECK: br label %loop

loop:
  %cmp1 = icmp sgt i32 %x, 0
  br i1 %cmp1, label %if, label %then

if:
  %add = add i32 %x, 1
  %cmp2 = icmp sgt i32 10, %add
  br i1 %cmp2, label %then, label %end

; CHECK-LABEL: then:
; CHECK-DISABLED: %phi = phi i32 [ %add, %if ], [ %x, %loop ]
then:
  %phi = phi i32 [ %add, %if ], [ %x, %loop ]
  store i32 %phi, i32* %p
  %cmp3 = icmp ne i32 %phi, 0
  br i1 %cmp3, label %loop, label %end

end:
  ret void
}

; CHECK-LABEL: @diamond_phi_oneloopexit
define void @diamond_phi_oneloopexit(i32 %x, i32* %p) {
; CHECK-LABEL: entry:
; CHECK-DAG: %add = add i32 %x, 1
; CHECK-DAG: %cmp1 = icmp sgt i32 %x, 0
; CHECK-DAG: %cmp2 = icmp sgt i32 10, %add
; CHECK-ENABLED: br i1 %cmp1, label %[[IF_LICM:.*]], label %[[THEN_LICM:.*]]
entry:
  br label %loop

; CHECK-ENABLED: [[IF_LICM]]:
; CHECK-ENABLED: br label %[[THEN_LICM:.*]]

; CHECK-ENABLED: [[ELSE_LICM]]:
; CHECK-DAG: %sub = sub i32 %x, 1
; CHECK-ENABLED: br label %[[THEN_LICM]]

; CHECK-ENABLED: [[THEN_LICM]]
; CHECK-ENABLED: %phi = phi i32 [ %add, %[[IF_LICM]] ], [ %sub, %[[ELSE_LICM]] ]
; CHECK-ENABLED: %cmp3 = icmp ne i32 %phi, 0
; CHECK: br label %loop

loop:
  %cmp1 = icmp sgt i32 %x, 0
  br i1 %cmp1, label %if, label %else

if:
  %add = add i32 %x, 1
  %cmp2 = icmp sgt i32 10, %add
  br i1 %cmp2, label %then, label %end

else:
  %sub = sub i32 %x, 1
  br label %then

; CHECK-LABEL: then:
; CHECK-DISABLED: %phi = phi i32 [ %add, %if ], [ %sub, %else ]
then:
  %phi = phi i32 [ %add, %if ], [ %sub, %else ]
  store i32 %phi, i32* %p
  %cmp3 = icmp ne i32 %phi, 0
  br i1 %cmp3, label %loop, label %end

end:
  ret void
}

; CHECK-LABEL: @diamond_phi_twoloopexit
define void @diamond_phi_twoloopexit(i32 %x, i32* %p) {
; CHECK-LABEL: entry:
; CHECK-DAG: %sub = sub i32 %x, 1
; CHECK-DAG: %add = add i32 %x, 1
; CHECK-DAG: %cmp1 = icmp sgt i32 %x, 0
; CHECK-DAG: %cmp2 = icmp sgt i32 10, %add
; CHECK-DAG: %cmp3 = icmp sgt i32 10, %sub
; CHECK-ENABLED: br i1 %cmp1, label %[[IF_LICM:.*]], label %[[THEN_LICM:.*]]
entry:
  br label %loop

; CHECK-ENABLED: [[IF_LICM]]:
; CHECK-ENABLED: br label %[[THEN_LICM:.*]]

; CHECK-ENABLED: [[ELSE_LICM]]:
; CHECK-ENABLED: br label %[[THEN_LICM]]

; CHECK-ENABLED: [[THEN_LICM]]
; CHECK-ENABLED: %phi = phi i32 [ %add, %[[IF_LICM]] ], [ %sub, %[[ELSE_LICM]] ]
; CHECK-ENABLED: %cmp4 = icmp ne i32 %phi, 0
; CHECK: br label %loop

loop:
  %cmp1 = icmp sgt i32 %x, 0
  br i1 %cmp1, label %if, label %else

if:
  %add = add i32 %x, 1
  %cmp2 = icmp sgt i32 10, %add
  br i1 %cmp2, label %then, label %end

else:
  %sub = sub i32 %x, 1
  %cmp3 = icmp sgt i32 10, %sub
  br i1 %cmp3, label %then, label %end

; CHECK-LABEL: then:
; CHECK-DISABLED: %phi = phi i32 [ %add, %if ], [ %sub, %else ]
; CHECK-DISABLED: %cmp4 = icmp ne i32 %phi, 0
then:
  %phi = phi i32 [ %add, %if ], [ %sub, %else ]
  store i32 %phi, i32* %p
  %cmp4 = icmp ne i32 %phi, 0
  br i1 %cmp4, label %loop, label %end

end:
  ret void
}

; The store cannot be hoisted, so add and shr cannot be hoisted into a
; conditional block.
; CHECK-LABEL: @conditional_use
define void @conditional_use(i32 %x, i32* %p) {
; CHECK-LABEL: entry:
; CHECK-DAG: %cond = icmp ugt i32 %x, 0
; CHECK-DAG: %add = add i32 %x, 5
; CHECK-DAG: %shr = ashr i32 %add, 1
; CHECK: br label %loop
entry:
  br label %loop

loop:
  %cond = icmp ugt i32 %x, 0
  br i1 %cond, label %if, label %else

; CHECK-LABEL: if:
; CHECK: store i32 %shr, i32* %p, align 4
if:
  %add = add i32 %x, 5
  %shr = ashr i32 %add, 1
  store i32 %shr, i32* %p, align 4
  br label %then

else:
  br label %then

then:
  br label %loop
}

; A diamond with two triangles on the left and one on the right. This test is
; to check that we have a unique loop preheader when we hoist the store (and so
; don't fail an assertion).
; CHECK-LABEL: @triangles_in_diamond
define void @triangles_in_diamond(i32* %ptr) {
; CHECK-LABEL: entry:
; CHECK: store i32 0, i32* %ptr, align 4
; CHECK: br label %loop
entry:
  br label %loop

loop:
  br i1 undef, label %left_triangle_1, label %right_triangle

left_triangle_1:
  br i1 undef, label %left_triangle_1_if, label %left_triangle_2

left_triangle_1_if:
  br label %left_triangle_2

left_triangle_2:
  br i1 undef, label %left_triangle_2_if, label %left_triangle_2_then

left_triangle_2_if:
  br label %left_triangle_2_then

left_triangle_2_then:
  br label %loop.end

right_triangle:
  br i1 undef, label %right_triangle.if, label %right_triangle.then

right_triangle.if:
  br label %right_triangle.then

right_triangle.then:
  br label %loop.end

loop.end:
  store i32 0, i32* %ptr, align 4
  br label %loop
}

; %cmp dominates its used after being hoisted, but not after %brmerge is rehoisted
; CHECK-LABEL: @rehoist
define void @rehoist(i8* %this, i32 %x) {
; CHECK-LABEL: entry:
; CHECK-DAG: %sub = add nsw i32 %x, -1
; CHECK-DAG: %fptr = bitcast i8* %this to void (i8*)*
; CHECK-DAG: %cmp = icmp eq i32 0, %sub
; CHECK-DAG: %brmerge = or i1 %cmp, true
entry:
  %sub = add nsw i32 %x, -1
  br label %loop

loop:
  br i1 undef, label %if1, label %else1

if1:
  %fptr = bitcast i8* %this to void (i8*)*
  call void %fptr(i8* %this)
  br label %then1

else1:
  br label %then1

then1:
  %cmp = icmp eq i32 0, %sub
  br i1 %cmp, label %end, label %else2

else2:
  %brmerge = or i1 %cmp, true
  br i1 %brmerge, label %if3, label %end

if3:
  br label %end

end:
  br label %loop
}

; A test case that uses empty blocks in a way that can cause control flow
; hoisting to get confused.
; CHECK-LABEL: @empty_blocks_multiple_conditional_branches
define void @empty_blocks_multiple_conditional_branches(float %arg, float* %ptr) {
; CHECK-LABEL: entry
; CHECK-DAG: %div1 = fmul float %arg, 4.000000e+00
; CHECK-DAG: %div2 = fmul float %arg, 2.000000e+00
entry:
  br label %loop

; The exact path to the phi isn't checked here, because it depends on whether
; cond2 or cond3 is hoisted first
; CHECK-ENABLED: %phi = phi float [ 0.000000e+00, %{{.*}} ], [ %div1, %{{.*}} ]
; CHECK: br label %loop

loop:
  br i1 undef, label %backedge2, label %cond1

cond1:
  br i1 undef, label %cond1.if, label %cond1.else

cond1.else:
  br label %cond3

cond1.if:
  br label %cond1.if.next

cond1.if.next:
  br label %cond2

cond2:
  %div1 = fmul float %arg, 4.000000e+00
  br i1 undef, label %cond2.if, label %cond2.then

cond2.if:
  br label %cond2.then

; CHECK-LABEL: cond2.then:
; CHECK-DISABLED: %phi = phi float [ 0.000000e+00, %cond2 ], [ %div1, %cond2.if ]
cond2.then:
  %phi = phi float [ 0.000000e+00, %cond2 ], [ %div1, %cond2.if ]
  store float %phi, float* %ptr
  br label %backedge2

cond3:
  br i1 undef, label %cond3.then, label %cond3.if

cond3.if:
  %div2 = fmul float %arg, 2.000000e+00
  store float %div2, float* %ptr
  br label %cond3.then

cond3.then:
  br label %loop

backedge2:
  br label %loop
}

; We can't do much here, so mainly just check that we don't crash.
; CHECK-LABEL: @many_path_phi
define void @many_path_phi(i32* %ptr1, i32* %ptr2) {
; CHECK-LABEL: entry:
; CHECK-DAG: %gep3 = getelementptr inbounds i32, i32* %ptr2, i32 2
; CHECK-DAG: %gep2 = getelementptr inbounds i32, i32* %ptr2, i32 2
; CHECK: br label %loop
entry:
  br label %loop

loop:
  %phi1 = phi i32 [ 0, %entry ], [ %phi2, %end ]
  %cmp1 = icmp ugt i32 %phi1, 3
  br i1 %cmp1, label %cond2, label %cond1

cond1:
  br i1 undef, label %end, label %cond1.else

cond1.else:
  %gep2 = getelementptr inbounds i32, i32* %ptr2, i32 2
  %val2 = load i32, i32* %gep2, align 4
  %cmp2 = icmp eq i32 %val2, 13
  br i1 %cmp2, label %cond1.end, label %end

cond1.end:
  br label %end

cond2:
  br i1 undef, label %end, label %cond2.else

cond2.else:
  %gep3 = getelementptr inbounds i32, i32* %ptr2, i32 2
  %val3 = load i32, i32* %gep3, align 4
  %cmp3 = icmp eq i32 %val3, 13
  br i1 %cmp3, label %cond2.end, label %end

cond2.end:
  br label %end

end:
  %phi2 = phi i32 [ 1, %cond1 ], [ 2, %cond1.else ], [ 3, %cond1.end ], [ 4, %cond2 ], [ 5, %cond2.else ], [ 6, %cond2.end ]
  br label %loop
}

; Check that we correctly handle the hoisting of %gep when theres a critical
; edge that branches to the preheader.
; CHECK-LABEL: @crit_edge
define void @crit_edge(i32* %ptr, i32 %idx, i1 %cond1, i1 %cond2) {
; CHECK-LABEL: entry:
; CHECK: %gep = getelementptr inbounds i32, i32* %ptr, i32 %idx
; CHECK: br label %preheader
entry:
  br label %preheader

preheader:
  br label %loop

loop:
  br i1 %cond1, label %then, label %if

if:
  %gep = getelementptr inbounds i32, i32* %ptr, i32 %idx
  %val = load i32, i32* %gep
  br label %then

then:
  %phi = phi i32 [ %val, %if ], [ 0, %loop ]
  store i32 %phi, i32* %ptr
  br i1 %cond2, label %loop, label %crit_edge

crit_edge:
  br label %preheader
}

; Check that the conditional sub is correctly hoisted from the inner loop to the
; preheader of the outer loop.
; CHECK-LABEL: @hoist_from_innermost_loop
define void @hoist_from_innermost_loop(i32 %nx, i32* %ptr) {
; CHECK-LABEL: entry:
; CHECK-DAG: %sub = sub nsw i32 0, %nx
; CHECK: br label %outer_loop
entry:
  br label %outer_loop

outer_loop:
  br label %middle_loop

middle_loop:
  br label %inner_loop

inner_loop:
  br i1 undef, label %inner_loop_end, label %if

if:
  %sub = sub nsw i32 0, %nx
  store i32 %sub, i32* %ptr, align 4
  br label %inner_loop_end

inner_loop_end:
  br i1 undef, label %inner_loop, label %middle_loop_end

middle_loop_end:
  br i1 undef, label %middle_loop, label %outer_loop_end

outer_loop_end:
  br label %outer_loop
}

; We have a diamond starting from %if, but %if.if is also reachable from %loop,
; so %gep should not be conditionally hoisted.
; CHECK-LABEL: @diamond_with_extra_in_edge
define void @diamond_with_extra_in_edge(i32* %ptr1, i32* %ptr2, i32 %arg) {
; CHECK-LABEL: entry:
; CHECK-DAG: %cmp2 = icmp ne i32 0, %arg
; CHECK-DAG: %gep = getelementptr i32, i32* %ptr1, i32 4
; CHECK: br label %loop
entry:
  br label %loop

loop:
  %phi1 = phi i32 [ 0, %entry ], [ %phi2, %then ]
  %cmp1 = icmp ugt i32 16, %phi1
  br i1 %cmp1, label %if, label %if.if

if:
  %cmp2 = icmp ne i32 0, %arg
  br i1 %cmp2, label %if.if, label %if.else

if.if:
  %gep = getelementptr i32, i32* %ptr1, i32 4
  %val = load i32, i32* %gep, align 4
  br label %then

if.else:
  br label %then

then:
  %phi2 = phi i32 [ %val, %if.if ], [ %phi1, %if.else ]
  store i32 %phi2, i32* %ptr2, align 4
  br label %loop
}

; %loop/%if/%then form a triangle, but %loop/%if/%then/%end also form a diamond.
; The triangle should be picked for conditional hoisting.
; CHECK-LABEL: @both_triangle_and_diamond
define void @both_triangle_and_diamond(i32* %ptr1, i32* %ptr2, i32 %arg) {
; CHECK-LABEL: entry:
; CHECK-DAG: %cmp1 = icmp ne i32 0, %arg
; CHECK-DAG: %gep = getelementptr i32, i32* %ptr1, i32 4
; CHECK-ENABLED: br i1 %cmp1, label %[[IF_LICM:.*]], label %[[THEN_LICM:.*]]
entry:
  br label %loop

; CHECK-ENABLED: [[IF_LICM]]:
; CHECK-ENABLED: br label %[[THEN_LICM]]

; CHECK-ENABLED: [[THEN_LICM]]:
; CHECK-ENABLED: %phi2 = phi i32 [ 0, %[[IF_LICM]] ], [ 1, %entry ]
; CHECK: br label %loop

loop:
  %phi1 = phi i32 [ 0, %entry ], [ %phi3, %end ]
  %cmp1 = icmp ne i32 0, %arg
  br i1 %cmp1, label %if, label %then

if:
  %gep = getelementptr i32, i32* %ptr1, i32 4
  %val = load i32, i32* %gep, align 4
  %cmp2 = icmp ugt i32 16, %phi1
  br i1 %cmp2, label %end, label %then

; CHECK-LABEL: then:
; CHECK-DISABLED: %phi2 = phi i32 [ 0, %if ], [ 1, %loop ]
then:
  %phi2 = phi i32 [ 0, %if ], [ 1, %loop ]
  br label %end

end:
  %phi3 = phi i32 [ %phi2, %then ], [ %val, %if ]
  store i32 %phi3, i32* %ptr2, align 4
  br label %loop
}

; We shouldn't duplicate the branch at the end of %loop and should instead hoist
; %val to %entry.
; CHECK-LABEL: @same_destination_branch
define i32 @same_destination_branch(i32 %arg1, i32 %arg2) {
; CHECK-LABEL: entry:
; CHECK-DAG: %cmp1 = icmp ne i32 %arg2, 0
; CHECK-DAG: %val = add i32 %arg1, 1
; CHECK: br label %loop
entry:
  br label %loop

; CHECK-LABEL: loop:
; CHECK: %phi = phi i32 [ 0, %entry ], [ %add, %then ]
loop:
  %phi = phi i32 [ 0, %entry ], [ %add, %then ]
  %add = add i32 %phi, 1
  %cmp1 = icmp ne i32 %arg2, 0
  br i1 %cmp1, label %if, label %if

if:
  %val = add i32 %arg1, 1
  br label %then

then:
  %cmp2 = icmp ne i32 %val, %phi
  br i1 %cmp2, label %loop, label %end

end:
  ret i32 %val
}

; Diamond-like control flow but the left/right blocks actually have the same
; destinations.
; TODO: We could potentially hoist all of phi2-4, but currently only hoist phi2.
; CHECK-LABEL: @diamond_like_same_destinations
define i32 @diamond_like_same_destinations(i32 %arg1, i32 %arg2) {
; CHECK-LABEL: entry:
; CHECK-DAG: %cmp1 = icmp ne i32 %arg1, 0
; CHECK-DAG: %cmp2 = icmp ugt i32 %arg2, 1
; CHECK-DAG: %cmp3 = icmp ugt i32 %arg2, 2
; CHECK-ENABLED: br i1 %cmp1, label %[[LEFT1_LICM:.*]], label %[[RIGHT1_LICM:.*]]
entry:
  br label %loop

; CHECK-ENABLED: [[LEFT1_LICM]]:
; CHECK-ENABLED: br label %[[LEFT2_LICM:.*]]

; CHECK-ENABLED: [[RIGHT1_LICM]]:
; CHECK-ENABLED: br label %[[LEFT2_LICM]]

; CHECK-ENABLED: [[LEFT2_LICM]]:
; CHECK-ENABLED: %phi2 = phi i32 [ 0, %[[LEFT1_LICM]] ], [ 1, %[[RIGHT1_LICM]] ]
; CHECK: br label %loop

loop:
  %phi1 = phi i32 [ 0, %entry ], [ %add, %loopend ]
  %add = add i32 %phi1, 1
  %cmp1 = icmp ne i32 %arg1, 0
  br i1 %cmp1, label %left1, label %right1

left1:
  %cmp2 = icmp ugt i32 %arg2, 1
  br i1 %cmp2, label %left2, label %right2

right1:
  %cmp3 = icmp ugt i32 %arg2, 2
  br i1 %cmp3, label %left2, label %right2

; CHECK-LABEL: left2:
; CHECK-DISABLED: %phi2 = phi i32 [ 0, %left1 ], [ 1, %right1 ]
left2:
  %phi2 = phi i32 [ 0, %left1 ], [ 1, %right1 ]
  br label %loopend

; CHECK-LABEL: right2:
; CHECK: %phi3 = phi i32 [ 2, %left1 ], [ 3, %right1 ]
right2:
  %phi3 = phi i32 [ 2, %left1 ], [ 3, %right1 ]
  br label %loopend

; CHECK-LABEL: loopend:
; CHECK: %phi4 = phi i32 [ %phi2, %left2 ], [ %phi3, %right2 ]
loopend:
  %phi4 = phi i32 [ %phi2, %left2 ], [ %phi3, %right2 ]
  %cmp4 = icmp ne i32 %phi1, 32
  br i1 %cmp4, label %loop, label %end

end:
  ret i32 %phi4
}

; A phi with multiple incoming values for the same block due to a branch with
; two destinations that are actually the same. We can't hoist this.
; TODO: This could be hoisted by erasing one of the incoming values.
; CHECK-LABEL: @phi_multiple_values_same_block
define i32 @phi_multiple_values_same_block(i32 %arg) {
; CHECK-LABEL: entry:
; CHECK: %cmp = icmp sgt i32 %arg, 4
; CHECK-NOT: phi
; CHECK: br label %loop
entry:
  br label %loop

loop:
  %cmp = icmp sgt i32 %arg, 4
  br i1 %cmp, label %if, label %then

if:
  br i1 undef, label %then, label %then

then:
  %phi = phi i32 [ %arg, %loop ], [ 1, %if ], [ 1, %if ]
  br i1 undef, label %exit, label %loop

exit:
  ret i32 %phi
}

; %phi is conditionally used in %d, and the store that %d is used in cannot be
; hoisted. This means that we have to rehoist %d, but have to make sure to
; rehoist it after %phi.
; CHECK-LABEL: @phi_conditional_use
define i64 @phi_conditional_use(i32 %f, i32* %g) {
; CHECK-LABEL: entry:
; CHECK: %cmp1 = icmp eq i32 %f, 1
; CHECK: %cmp2 = icmp eq i32 %f, 0
; CHECK-ENABLED: br i1 %cmp1, label %[[IF_END_LICM:.*]], label %[[IF_THEN_LICM:.*]]
entry:
  %cmp1 = icmp eq i32 %f, 1
  %cmp2 = icmp eq i32 %f, 0
  br label %loop

; CHECK-ENABLED: [[IF_THEN_LICM]]:
; CHECK-ENABLED: br label %[[IF_END_LICM]]

; CHECK-ENABLED: [[IF_END_LICM]]:
; CHECK-ENABLED: %phi = phi i64 [ 0, %entry ], [ 1, %[[IF_THEN_LICM]] ]
; CHECK-ENABLED: %d = getelementptr inbounds i32, i32* %g, i64 %phi
; CHECK-ENABLED: i1 %cmp2, label %[[LOOP_BACKEDGE_LICM:.*]], label %[[IF_THEN2_LICM:.*]]

; CHECK-ENABLED: [[IF_THEN2_LICM]]:
; CHECK-ENABLED: br label %[[LOOP_BACKEDGE_LICM]]

; CHECK-ENABLED: [[LOOP_BACKEDGE_LICM]]:
; CHECK: br label %loop

loop:
  br i1 %cmp1, label %if.end, label %if.then

if.then:
  br label %if.end

; CHECK-LABEL: if.end:
; CHECK-DISABLED: %phi = phi i64 [ 0, %loop ], [ 1, %if.then ]
if.end:
  %phi = phi i64 [ 0, %loop ], [ 1, %if.then ]
  br i1 %cmp2, label %loop.backedge, label %if.then2

; CHECK-LABEL: if.then2:
; CHECK-DISABLED: %d = getelementptr inbounds i32, i32* %g, i64 %phi
if.then2:
  %d = getelementptr inbounds i32, i32* %g, i64 %phi
  store i32 1, i32* %d, align 4
  br label %loop.backedge

loop.backedge:
  br label %loop
}

; As above, but we have two such phis
; CHECK-LABEL: @phi_conditional_use_twice
define i64 @phi_conditional_use_twice(i32 %f, i32* %g) {
; CHECK-LABEL: entry:
; CHECK: %cmp1 = icmp eq i32 %f, 1
; CHECK: %cmp2 = icmp eq i32 %f, 0
; CHECK-ENABLED: br i1 %cmp1, label %[[IF_END_LICM:.*]], label %[[IF_THEN_LICM:.*]]
entry:
  %cmp1 = icmp eq i32 %f, 1
  %cmp2 = icmp eq i32 %f, 0
  %cmp3 = icmp sgt i32 %f, 0
  br label %loop

; CHECK-ENABLED: [[IF_THEN_LICM]]:
; CHECK-ENABLED: br label %[[IF_END_LICM]]

; CHECK-ENABLED: [[IF_END_LICM]]:
; CHECK-ENABLED: %phi1 = phi i64 [ 0, %entry ], [ 1, %[[IF_THEN_LICM]] ]
; CHECK-ENABLED: %d = getelementptr inbounds i32, i32* %g, i64 %phi1
; CHECK-ENABLED: i1 %cmp2, label %[[IF_END2_LICM:.*]], label %[[IF_THEN2_LICM:.*]]

; CHECK-ENABLED: [[IF_THEN2_LICM]]:
; CHECK-ENABLED: br label %[[IF_END2_LICM]]

; CHECK-ENABLED: [[IF_END2_LICM]]:
; CHECK-ENABLED: %phi2 = phi i64 [ 2, %[[IF_END_LICM]] ], [ 3, %[[IF_THEN2_LICM]] ]
; CHECK-ENABLED: %e = getelementptr inbounds i32, i32* %g, i64 %phi2
; CHECK-ENABLED: i1 %cmp3, label %[[LOOP_BACKEDGE_LICM:.*]], label %[[IF_THEN3_LICM:.*]]

; CHECK-ENABLED: [[IF_THEN3_LICM]]:
; CHECK-ENABLED: br label %[[LOOP_BACKEDGE_LICM]]

; CHECK-ENABLED: [[LOOP_BACKEDGE_LICM]]:
; CHECK: br label %loop

loop:
  br i1 %cmp1, label %if.end, label %if.then

if.then:
  br label %if.end

; CHECK-LABEL: if.end:
; CHECK-DISABLED: %phi1 = phi i64 [ 0, %loop ], [ 1, %if.then ]
if.end:
  %phi1 = phi i64 [ 0, %loop ], [ 1, %if.then ]
  br i1 %cmp2, label %if.end2, label %if.then2

; CHECK-LABEL: if.then2:
; CHECK-DISABLED: %d = getelementptr inbounds i32, i32* %g, i64 %phi1
if.then2:
  %d = getelementptr inbounds i32, i32* %g, i64 %phi1
  store i32 1, i32* %d, align 4
  br label %if.end2

; CHECK-LABEL: if.end2:
; CHECK-DISABLED: %phi2 = phi i64 [ 2, %if.end ], [ 3, %if.then2 ]
if.end2:
  %phi2 = phi i64 [ 2, %if.end ], [ 3, %if.then2 ]
  br i1 %cmp3, label %loop.backedge, label %if.then3

; CHECK-LABEL: if.then3:
; CHECK-DISABLED: %e = getelementptr inbounds i32, i32* %g, i64 %phi2
if.then3:
  %e = getelementptr inbounds i32, i32* %g, i64 %phi2
  store i32 1, i32* %e, align 4
  br label %loop.backedge

loop.backedge:
  br label %loop
}

; The order that we hoist instructions from the loop is different to the textual
; order in the function. Check that we can rehoist this correctly.
; CHECK-LABEL: @rehoist_wrong_order_1
define void @rehoist_wrong_order_1(i32* %ptr) {
; CHECK-LABEL: entry
; CHECK-DAG: %gep2 = getelementptr inbounds i32, i32* %ptr, i64 2
; CHECK-DAG: %gep3 = getelementptr inbounds i32, i32* %ptr, i64 3
; CHECK-DAG: %gep1 = getelementptr inbounds i32, i32* %ptr, i64 1
; CHECK-ENABLED: br i1 undef, label %[[IF1_LICM:.*]], label %[[ELSE1_LICM:.*]]
entry:
  br label %loop

; CHECK-ENABLED: [[IF1_LICM]]:
; CHECK-ENABLED: br label %[[LOOP_BACKEDGE_LICM:.*]]

; CHECK-ENABLED: [[ELSE1_LICM]]:
; CHECK-ENABLED: br label %[[LOOP_BACKEDGE_LICM]]

; CHECK-ENABLED: [[LOOP_BACKEDGE_LICM]]:
; CHECK-ENABLED: br i1 undef, label %[[IF3_LICM:.*]], label %[[END_LICM:.*]]

; CHECK-ENABLED: [[IF3_LICM]]:
; CHECK-ENABLED: br label %[[END_LICM]]

; CHECK-ENABLED: [[END_LICM]]:
; CHECK: br label %loop

loop:
  br i1 undef, label %if1, label %else1

if1:
  %gep1 = getelementptr inbounds i32, i32* %ptr, i64 1
  store i32 0, i32* %gep1, align 4
  br label %loop.backedge

else1:
  %gep2 = getelementptr inbounds i32, i32* %ptr, i64 2
  store i32 0, i32* %gep2, align 4
  br i1 undef, label %if2, label %loop.backedge

if2:
  br i1 undef, label %if3, label %end

if3:
  %gep3 = getelementptr inbounds i32, i32* %ptr, i64 3
  store i32 0, i32* %gep3, align 4
  br label %end

end:
  br label %loop.backedge

loop.backedge:
  br label %loop

}

; CHECK-LABEL: @rehoist_wrong_order_2
define void @rehoist_wrong_order_2(i32* %ptr) {
; CHECK-LABEL: entry
; CHECK-DAG: %gep2 = getelementptr inbounds i32, i32* %ptr, i64 2
; CHECK-DAG: %gep3 = getelementptr inbounds i32, i32* %gep2, i64 3
; CHECK-DAG: %gep1 = getelementptr inbounds i32, i32* %ptr, i64 1
; CHECK-ENABLED: br i1 undef, label %[[IF1_LICM:.*]], label %[[ELSE1_LICM:.*]]
entry:
  br label %loop

; CHECK-ENABLED: [[IF1_LICM]]:
; CHECK-ENABLED: br label %[[LOOP_BACKEDGE_LICM:.*]]

; CHECK-ENABLED: [[ELSE1_LICM]]:
; CHECK-ENABLED: br label %[[LOOP_BACKEDGE_LICM]]

; CHECK-ENABLED: [[LOOP_BACKEDGE_LICM]]:
; CHECK-ENABLED: br i1 undef, label %[[IF3_LICM:.*]], label %[[END_LICM:.*]]

; CHECK-ENABLED: [[IF3_LICM]]:
; CHECK-ENABLED: br label %[[END_LICM]]

; CHECK-ENABLED: [[END_LICM]]:
; CHECK: br label %loop

loop:
  br i1 undef, label %if1, label %else1

if1:
  %gep1 = getelementptr inbounds i32, i32* %ptr, i64 1
  store i32 0, i32* %gep1, align 4
  br label %loop.backedge

else1:
  %gep2 = getelementptr inbounds i32, i32* %ptr, i64 2
  store i32 0, i32* %gep2, align 4
  br i1 undef, label %if2, label %loop.backedge

if2:
  br i1 undef, label %if3, label %end

if3:
  %gep3 = getelementptr inbounds i32, i32* %gep2, i64 3
  store i32 0, i32* %gep3, align 4
  br label %end

end:
  br label %loop.backedge

loop.backedge:
  br label %loop
}

; CHECK-LABEL: @rehoist_wrong_order_3
define void @rehoist_wrong_order_3(i32* %ptr) {
; CHECK-LABEL: entry
; CHECK-DAG: %gep2 = getelementptr inbounds i32, i32* %ptr, i64 2
; CHECK-DAG: %gep1 = getelementptr inbounds i32, i32* %ptr, i64 1
; CHECK-ENABLED: br i1 undef, label %[[IF1_LICM:.*]], label %[[ELSE1_LICM:.*]]
entry:
  br label %loop

; CHECK-ENABLED: [[IF1_LICM]]:
; CHECK-ENABLED: br label %[[IF2_LICM:.*]]

; CHECK-ENABLED: [[ELSE1_LICM]]:
; CHECK-ENABLED: br label %[[IF2_LICM]]

; CHECK-ENABLED: [[IF2_LICM]]:
; CHECK-ENABLED: %phi = phi i32* [ %gep1, %[[IF1_LICM]] ], [ %gep2, %[[ELSE1_LICM]] ]
; CHECK-ENABLED: %gep3 = getelementptr inbounds i32, i32* %phi, i64 3
; CHECK-ENABLED: br i1 undef, label %[[IF3_LICM:.*]], label %[[END_LICM:.*]]

; CHECK-ENABLED: [[IF3_LICM]]:
; CHECK-ENABLED: br label %[[END_LICM]]

; CHECK-ENABLED: [[END_LICM]]:
; CHECK: br label %loop

loop:
  br i1 undef, label %if1, label %else1

if1:
  %gep1 = getelementptr inbounds i32, i32* %ptr, i64 1
  store i32 0, i32* %gep1, align 4
  br label %if2

else1:
  %gep2 = getelementptr inbounds i32, i32* %ptr, i64 2
  store i32 0, i32* %gep2, align 4
  br i1 undef, label %if2, label %loop.backedge

if2:
  %phi = phi i32* [ %gep1, %if1 ], [ %gep2, %else1 ]
  br i1 undef, label %if3, label %end

if3:
  %gep3 = getelementptr inbounds i32, i32* %phi, i64 3
  store i32 0, i32* %gep3, align 4
  br label %end

end:
  br label %loop.backedge

loop.backedge:
  br label %loop
}
