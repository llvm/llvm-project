; RUN: opt -inline -S %s | FileCheck %s

declare void @g()


;;; Test with a call in a funclet that needs to remain a call
;;; when inlined because the funclet doesn't unwind to caller.
;;; CHECK-LABEL: define void @test1(
define void @test1() personality void ()* @g {
entry:
; CHECK-NEXT: entry:
  invoke void @test1_inlinee()
    to label %exit unwind label %cleanup
cleanup:
  %pad = cleanuppad within none []
  call void @g() [ "funclet"(token %pad) ]
  cleanupret from %pad unwind to caller
exit:
  ret void
}

define void @test1_inlinee() alwaysinline personality void ()* @g {
entry:
  invoke void @g()
    to label %exit unwind label %cleanup.inner
; CHECK-NEXT:  invoke void @g()
; CHECK-NEXT:    unwind label %[[cleanup_inner:.+]]

cleanup.inner:
  %pad.inner = cleanuppad within none []
  call void @g() [ "funclet"(token %pad.inner) ]
  cleanupret from %pad.inner unwind label %cleanup.outer
; CHECK: [[cleanup_inner]]:
; The call here needs to remain a call becuase pad.inner has a cleanupret
; that stays within the inlinee.
; CHECK-NEXT:  %[[pad_inner:[^ ]+]] = cleanuppad within none
; CHECK-NEXT:  call void @g() [ "funclet"(token %[[pad_inner]]) ]
; CHECK-NEXT:  cleanupret from %[[pad_inner]] unwind label %[[cleanup_outer:.+]]

cleanup.outer:
  %pad.outer = cleanuppad within none []
  call void @g() [ "funclet"(token %pad.outer) ]
  cleanupret from %pad.outer unwind to caller
; CHECK: [[cleanup_outer]]:
; The call and cleanupret here need to be redirected to caller cleanup
; CHECK-NEXT: %[[pad_outer:[^ ]+]] = cleanuppad within none
; CHECK-NEXT: invoke void @g() [ "funclet"(token %[[pad_outer]]) ]
; CHECK-NEXT:   unwind label %cleanup
; CHECK: cleanupret from %[[pad_outer]] unwind label %cleanup{{$}}

exit:
  ret void
}



;;; Test with an "unwind to caller" catchswitch in a parent funclet
;;; that needs to remain "unwind to caller" because the parent
;;; doesn't unwind to caller.
;;; CHECK-LABEL: define void @test2(
define void @test2() personality void ()* @g {
entry:
; CHECK-NEXT: entry:
  invoke void @test2_inlinee()
    to label %exit unwind label %cleanup
cleanup:
  %pad = cleanuppad within none []
  call void @g() [ "funclet"(token %pad) ]
  cleanupret from %pad unwind to caller
exit:
  ret void
}

define void @test2_inlinee() alwaysinline personality void ()* @g {
entry:
  invoke void @g()
    to label %exit unwind label %cleanup1
; CHECK-NEXT:   invoke void @g()
; CHECK-NEXT:     unwind label %[[cleanup1:.+]]

cleanup1:
  %outer = cleanuppad within none []
  invoke void @g() [ "funclet"(token %outer) ]
    to label %ret1 unwind label %catchswitch
; CHECK: [[cleanup1]]:
; CHECK-NEXT: %[[outer:[^ ]+]] = cleanuppad within none
; CHECK-NEXT: invoke void @g() [ "funclet"(token %[[outer]]) ]
; CHECK-NEXT:   unwind label %[[catchswitch:.+]]

catchswitch:
  %cs = catchswitch within %outer [label %catch] unwind to caller
; CHECK: [[catchswitch]]:
; The catchswitch here needs to remain "unwind to caller" since %outer
; has a cleanupret that remains within the inlinee.
; CHECK-NEXT: %[[cs:[^ ]+]] = catchswitch within %[[outer]] [label %[[catch:.+]]] unwind to caller

catch:
  %inner = catchpad within %cs []
  call void @g() [ "funclet"(token %inner) ]
  catchret from %inner to label %ret1
; CHECK: [[catch]]:
; The call here needs to remain a call since it too is within %outer
; CHECK:   %[[inner:[^ ]+]] = catchpad within %[[cs]]
; CHECK-NEXT: call void @g() [ "funclet"(token %[[inner]]) ]

ret1:
  cleanupret from %outer unwind label %cleanup2
; CHECK: cleanupret from %[[outer]] unwind label %[[cleanup2:.+]]

cleanup2:
  %later = cleanuppad within none []
  cleanupret from %later unwind to caller
; CHECK: [[cleanup2]]:
; The cleanupret here needs to get redirected to the caller cleanup
; CHECK-NEXT: %[[later:[^ ]+]] = cleanuppad within none
; CHECK-NEXT: cleanupret from %[[later]] unwind label %cleanup{{$}}

exit:
  ret void
}


;;; Test with a call in a cleanup that has no definitive unwind
;;; destination, that must be rewritten to an invoke.
;;; CHECK-LABEL: define void @test3(
define void @test3() personality void ()* @g {
entry:
; CHECK-NEXT: entry:
  invoke void @test3_inlinee()
    to label %exit unwind label %cleanup
cleanup:
  %pad = cleanuppad within none []
  call void @g() [ "funclet"(token %pad) ]
  cleanupret from %pad unwind to caller
exit:
  ret void
}

define void @test3_inlinee() alwaysinline personality void ()* @g {
entry:
  invoke void @g()
    to label %exit unwind label %cleanup
; CHECK-NEXT:  invoke void @g()
; CHECK-NEXT:    unwind label %[[cleanup:.+]]

cleanup:
  %pad = cleanuppad within none []
  call void @g() [ "funclet"(token %pad) ]
  unreachable
; CHECK: [[cleanup]]:
; The call must be rewritten to an invoke targeting the caller cleanup
; because it may well unwind to there.
; CHECK-NEXT: %[[pad:[^ ]+]] = cleanuppad within none
; CHECK-NEXT: invoke void @g() [ "funclet"(token %[[pad]]) ]
; CHECK-NEXT:   unwind label %cleanup{{$}}

exit:
  ret void
}


;;; Test with a catchswitch in a cleanup that has no definitive
;;; unwind destination, that must be rewritten to unwind to the
;;; inlined invoke's unwind dest
;;; CHECK-LABEL: define void @test4(
define void @test4() personality void ()* @g {
entry:
; CHECK-NEXT: entry:
  invoke void @test4_inlinee()
    to label %exit unwind label %cleanup
cleanup:
  %pad = cleanuppad within none []
  call void @g() [ "funclet"(token %pad) ]
  cleanupret from %pad unwind to caller
exit:
  ret void
}

define void @test4_inlinee() alwaysinline personality void ()* @g {
entry:
  invoke void @g()
    to label %exit unwind label %cleanup
; CHECK-NEXT: invoke void @g()
; CHECK-NEXT:   unwind label %[[cleanup:.+]]

cleanup:
  %clean = cleanuppad within none []
  invoke void @g() [ "funclet"(token %clean) ]
    to label %unreachable unwind label %dispatch
; CHECK: [[cleanup]]:
; CHECK-NEXT: %[[clean:[^ ]+]] = cleanuppad within none
; CHECK-NEXT: invoke void @g() [ "funclet"(token %[[clean]]) ]
; CHECK-NEXT:   unwind label %[[dispatch:.+]]

dispatch:
  %cs = catchswitch within %clean [label %catch] unwind to caller
; CHECK: [[dispatch]]:
; The catchswitch must be rewritten to unwind to %cleanup in the caller
; because it may well unwind to there.
; CHECK-NEXT: %[[cs:[^ ]+]] = catchswitch within %[[clean]] [label %[[catch:.+]]] unwind label %cleanup{{$}}

catch:
  catchpad within %cs []
  br label %unreachable
unreachable:
  unreachable
exit:
  ret void
}


;;; Test with multiple levels of nesting, and unwind dests
;;; that need to be inferred from ancestors, descendants,
;;; and cousins.
;;; CHECK-LABEL: define void @test5(
define void @test5() personality void ()* @g {
entry:
; CHECK-NEXT: entry:
  invoke void @test5_inlinee()
    to label %exit unwind label %cleanup
cleanup:
  %pad = cleanuppad within none []
  call void @g() [ "funclet"(token %pad) ]
  cleanupret from %pad unwind to caller
exit:
  ret void
}

define void @test5_inlinee() alwaysinline personality void ()* @g {
entry:
  invoke void @g()
    to label %cont unwind label %noinfo.root
; CHECK-NEXT: invoke void @g()
; CHECK-NEXT:   to label %[[cont:[^ ]+]] unwind label %[[noinfo_root:.+]]

noinfo.root:
  %noinfo.root.pad = cleanuppad within none []
  call void @g() [ "funclet"(token %noinfo.root.pad) ]
  invoke void @g() [ "funclet"(token %noinfo.root.pad) ]
    to label %noinfo.root.cont unwind label %noinfo.left
; CHECK: [[noinfo_root]]:
; Nothing under "noinfo.root" has a definitive unwind destination, so
; we must assume all of it may actually unwind, and redirect unwinds
; to the cleanup in the caller.
; CHECK-NEXT: %[[noinfo_root_pad:[^ ]+]] = cleanuppad within none []
; CHECK-NEXT: invoke void @g() [ "funclet"(token %[[noinfo_root_pad]]) ]
; CHECK-NEXT:   to label %[[next:[^ ]+]] unwind label %cleanup{{$}}
; CHECK: [[next]]:
; CHECK-NEXT: invoke void @g() [ "funclet"(token %[[noinfo_root_pad]]) ]
; CHECK-NEXT:   to label %[[noinfo_root_cont:[^ ]+]] unwind label %[[noinfo_left:.+]]

noinfo.left:
  %noinfo.left.pad = cleanuppad within %noinfo.root.pad []
  invoke void @g() [ "funclet"(token %noinfo.left.pad) ]
    to label %unreachable unwind label %noinfo.left.child
; CHECK: [[noinfo_left]]:
; CHECK-NEXT: %[[noinfo_left_pad:[^ ]+]] = cleanuppad within %[[noinfo_root_pad]]
; CHECK-NEXT: invoke void @g() [ "funclet"(token %[[noinfo_left_pad]]) ]
; CHECK-NEXT:   unwind label %[[noinfo_left_child:.+]]

noinfo.left.child:
  %noinfo.left.child.cs = catchswitch within %noinfo.left.pad [label %noinfo.left.child.catch] unwind to caller
; CHECK: [[noinfo_left_child]]:
; CHECK-NEXT: %[[noinfo_left_child_cs:[^ ]+]] = catchswitch within %[[noinfo_left_pad]] [label %[[noinfo_left_child_catch:[^ ]+]]] unwind label %cleanup{{$}}

noinfo.left.child.catch:
  %noinfo.left.child.pad = catchpad within %noinfo.left.child.cs []
  call void @g() [ "funclet"(token %noinfo.left.child.pad) ]
  br label %unreachable
; CHECK: [[noinfo_left_child_catch]]:
; CHECK-NEXT: %[[noinfo_left_child_pad:[^ ]+]] = catchpad within %[[noinfo_left_child_cs]] []
; CHECK-NEXT: invoke void @g() [ "funclet"(token %[[noinfo_left_child_pad]]) ]
; CHECK-NEXT:   unwind label %cleanup{{$}}

noinfo.root.cont:
  invoke void @g() [ "funclet"(token %noinfo.root.pad) ]
    to label %unreachable unwind label %noinfo.right
; CHECK: [[noinfo_root_cont]]:
; CHECK-NEXT: invoke void @g() [ "funclet"(token %[[noinfo_root_pad]]) ]
; CHECK-NEXT:   unwind label %[[noinfo_right:.+]]

noinfo.right:
  %noinfo.right.cs = catchswitch within %noinfo.root.pad [label %noinfo.right.catch] unwind to caller
; CHECK: [[noinfo_right]]:
; CHECK-NEXT: %[[noinfo_right_cs:[^ ]+]] = catchswitch within %[[noinfo_root_pad]] [label %[[noinfo_right_catch:[^ ]+]]] unwind label %cleanup{{$}}

noinfo.right.catch:
  %noinfo.right.pad = catchpad within %noinfo.right.cs []
  invoke void @g() [ "funclet"(token %noinfo.right.pad) ]
    to label %unreachable unwind label %noinfo.right.child
; CHECK: [[noinfo_right_catch]]:
; CHECK-NEXT: %[[noinfo_right_pad:[^ ]+]] = catchpad within %[[noinfo_right_cs]]
; CHECK-NEXT: invoke void @g() [ "funclet"(token %[[noinfo_right_pad]]) ]
; CHECK-NEXT:   unwind label %[[noinfo_right_child:.+]]

noinfo.right.child:
  %noinfo.right.child.pad = cleanuppad within %noinfo.right.pad []
  call void @g() [ "funclet"(token %noinfo.right.child.pad) ]
  br label %unreachable
; CHECK: [[noinfo_right_child]]:
; CHECK-NEXT: %[[noinfo_right_child_pad:[^ ]+]] = cleanuppad within %[[noinfo_right_pad]]
; CHECK-NEXT: invoke void @g() [ "funclet"(token %[[noinfo_right_child_pad]]) ]
; CHECK-NEXT:   unwind label %cleanup{{$}}

cont:
  invoke void @g()
    to label %exit unwind label %implicit.root
; CHECK: [[cont]]:
; CHECK-NEXT: invoke void @g()
; CHECK-NEXT:   unwind label %[[implicit_root:.+]]

implicit.root:
  %implicit.root.pad = cleanuppad within none []
  call void @g() [ "funclet"(token %implicit.root.pad) ]
  invoke void @g() [ "funclet"(token %implicit.root.pad) ]
    to label %implicit.root.cont unwind label %implicit.left
; CHECK: [[implicit_root]]:
; There's an unwind edge to %internal in implicit.right, and we need to propagate that
; fact down to implicit.right.grandchild, up to implicit.root, and down to
; implicit.left.child.catch, leaving all calls and "unwind to caller" catchswitches
; alone to so they don't conflict with the unwind edge in implicit.right
; CHECK-NEXT: %[[implicit_root_pad:[^ ]+]] = cleanuppad within none
; CHECK-NEXT: call void @g() [ "funclet"(token %[[implicit_root_pad]]) ]
; CHECK-NEXT: invoke void @g() [ "funclet"(token %[[implicit_root_pad]]) ]
; CHECK-NEXT:   to label %[[implicit_root_cont:[^ ]+]] unwind label %[[implicit_left:.+]]

implicit.left:
  %implicit.left.pad = cleanuppad within %implicit.root.pad []
  invoke void @g() [ "funclet"(token %implicit.left.pad) ]
    to label %unreachable unwind label %implicit.left.child
; CHECK: [[implicit_left]]:
; CHECK-NEXT: %[[implicit_left_pad:[^ ]+]] = cleanuppad within %[[implicit_root_pad:[^ ]+]]
; CHECK-NEXT: invoke void @g() [ "funclet"(token %[[implicit_left_pad]]) ]
; CHECK-NEXT:   unwind label %[[implicit_left_child:.+]]

implicit.left.child:
  %implicit.left.child.cs = catchswitch within %implicit.left.pad [label %implicit.left.child.catch] unwind to caller
; CHECK: [[implicit_left_child]]:
; CHECK-NEXT: %[[implicit_left_child_cs:[^ ]+]] = catchswitch within %[[implicit_left_pad]] [label %[[implicit_left_child_catch:[^ ]+]]] unwind to caller

implicit.left.child.catch:
  %implicit.left.child.pad = catchpad within %implicit.left.child.cs []
  call void @g() [ "funclet"(token %implicit.left.child.pad) ]
  br label %unreachable
; CHECK: [[implicit_left_child_catch]]:
; CHECK-NEXT: %[[implicit_left_child_pad:[^ ]+]] = catchpad within %[[implicit_left_child_cs]]
; CHECK-NEXT: call void @g() [ "funclet"(token %[[implicit_left_child_pad]]) ]

implicit.root.cont:
  invoke void @g() [ "funclet"(token %implicit.root.pad) ]
    to label %unreachable unwind label %implicit.right
; CHECK: [[implicit_root_cont]]:
; CHECK-NEXT: invoke void @g() [ "funclet"(token %[[implicit_root_pad]]) ]
; CHECK-NEXT:   unwind label %[[implicit_right:.+]]

implicit.right:
  %implicit.right.cs = catchswitch within %implicit.root.pad [label %implicit.right.catch] unwind label %internal
; CHECK: [[implicit_right]]:
; This is the unwind edge (to %internal) whose existence needs to get propagated around the "implicit" tree
; CHECK-NEXT: %[[implicit_right_cs:[^ ]+]] = catchswitch within %[[implicit_root_pad]] [label %[[implicit_right_catch:[^ ]+]]] unwind label %[[internal:.+]]

implicit.right.catch:
  %implicit.right.pad = catchpad within %implicit.right.cs []
  invoke void @g() [ "funclet"(token %implicit.right.pad) ]
    to label %unreachable unwind label %implicit.right.child
; CHECK: [[implicit_right_catch]]:
; CHECK-NEXT: %[[implicit_right_pad:[^ ]+]] = catchpad within %[[implicit_right_cs]]
; CHECK-NEXT: invoke void @g() [ "funclet"(token %[[implicit_right_pad]]) ]
; CHECK-NEXT:   unwind label %[[implicit_right_child:.+]]

implicit.right.child:
  %implicit.right.child.pad = cleanuppad within %implicit.right.pad []
  invoke void @g() [ "funclet"(token %implicit.right.child.pad) ]
    to label %unreachable unwind label %implicit.right.grandchild
; CHECK: [[implicit_right_child]]:
; CHECK-NEXT: %[[implicit_right_child_pad:[^ ]+]] = cleanuppad within %[[implicit_right_pad]]
; CHECK-NEXT: invoke void @g() [ "funclet"(token %[[implicit_right_child_pad]]) ]
; CHECK-NEXT:   unwind label %[[implicit_right_grandchild:.+]]

implicit.right.grandchild:
  %implicit.right.grandchild.cs = catchswitch within %implicit.right.child.pad [label %implicit.right.grandchild.catch] unwind to caller
; CHECK: [[implicit_right_grandchild]]:
; CHECK-NEXT: %[[implicit_right_grandchild_cs:[^ ]+]] = catchswitch within %[[implicit_right_child_pad]] [label %[[implicit_right_grandchild_catch:[^ ]+]]] unwind to caller

implicit.right.grandchild.catch:
  %implicit.right.grandhcild.pad = catchpad within %implicit.right.grandchild.cs []
  call void @g() [ "funclet"(token %implicit.right.grandhcild.pad) ]
  br label %unreachable
; CHECK: [[implicit_right_grandchild_catch]]:
; CHECK-NEXT: %[[implicit_right_grandhcild_pad:[^ ]+]] = catchpad within %[[implicit_right_grandchild_cs]]
; CHECK-NEXT: call void @g() [ "funclet"(token %[[implicit_right_grandhcild_pad]]) ]

internal:
  %internal.pad = cleanuppad within none []
  call void @g() [ "funclet"(token %internal.pad) ]
  cleanupret from %internal.pad unwind to caller
; CHECK: [[internal]]:
; internal is a cleanup with a "return to caller" cleanuppad; that needs to get redirected
; to %cleanup in the caller, and the call needs to get similarly rewritten to an invoke.
; CHECK-NEXT: %[[internal_pad:[^ ]+]] = cleanuppad within none
; CHECK-NEXT: invoke void @g() [ "funclet"(token %internal.pad.i) ]
; CHECK-NEXT:   to label %[[next:[^ ]+]] unwind label %cleanup{{$}}
; CHECK: [[next]]:
; CHECK-NEXT: cleanupret from %[[internal_pad]] unwind label %cleanup{{$}}

unreachable:
  unreachable
exit:
  ret void
}

;;; Test with funclets that don't have information for themselves, but have
;;; descendants which unwind to other descendants (left.left unwinds to
;;; left.right, and right unwinds to far_right).  Make sure that these local
;;; unwinds don't trip up processing of the ancestor nodes (left and root) that
;;; ultimately have no information.
;;; CHECK-LABEL: define void @test6(
define void @test6() personality void()* @ProcessCLRException {
entry:
; CHECK-NEXT: entry:
  invoke void @test6_inlinee()
    to label %exit unwind label %cleanup
cleanup:
  %pad = cleanuppad within none []
  call void @g() [ "funclet"(token %pad) ]
  cleanupret from %pad unwind to caller
exit:
  ret void
}

define void @test6_inlinee() alwaysinline personality void ()* @ProcessCLRException {
entry:
  invoke void @g()
    to label %exit unwind label %root
    ; CHECK-NEXT:  invoke void @g()
    ; CHECK-NEXT:    unwind label %[[root:.+]]
root:
  %root.pad = cleanuppad within none []
  invoke void @g() [ "funclet"(token %root.pad) ]
    to label %root.cont unwind label %left
; CHECK: [[root]]:
; CHECK-NEXT: %[[root_pad:.+]] = cleanuppad within none []
; CHECK-NEXT: invoke void @g() [ "funclet"(token %[[root_pad]]) ]
; CHECK-NEXT:   to label %[[root_cont:.+]] unwind label %[[left:.+]]

left:
  %left.cs = catchswitch within %root.pad [label %left.catch] unwind to caller
; CHECK: [[left]]:
; CHECK-NEXT: %[[left_cs:.+]] = catchswitch within %[[root_pad]] [label %[[left_catch:.+]]] unwind label %cleanup

left.catch:
  %left.cp = catchpad within %left.cs []
  call void @g() [ "funclet"(token %left.cp) ]
  invoke void @g() [ "funclet"(token %left.cp) ]
    to label %unreach unwind label %left.left
; CHECK: [[left_catch:.+]]:
; CHECK-NEXT: %[[left_cp:.+]] = catchpad within %[[left_cs]] []
; CHECK-NEXT: invoke void @g() [ "funclet"(token %[[left_cp]]) ]
; CHECK-NEXT:   to label %[[lc_cont:.+]] unwind label %cleanup
; CHECK: [[lc_cont]]:
; CHECK-NEXT: invoke void @g() [ "funclet"(token %[[left_cp]]) ]
; CHECK-NEXT:   to label %[[unreach:.+]] unwind label %[[left_left:.+]]

left.left:
  %ll.pad = cleanuppad within %left.cp []
  cleanupret from %ll.pad unwind label %left.right
; CHECK: [[left_left]]:
; CHECK-NEXT: %[[ll_pad:.+]] = cleanuppad within %[[left_cp]] []
; CHECK-NEXT: cleanupret from %[[ll_pad]] unwind label %[[left_right:.+]]

left.right:
  %lr.pad = cleanuppad within %left.cp []
  unreachable
; CHECK: [[left_right]]:
; CHECK-NEXT: %[[lr_pad:.+]] = cleanuppad within %[[left_cp]] []
; CHECK-NEXT: unreachable

root.cont:
  call void @g() [ "funclet"(token %root.pad) ]
  invoke void @g() [ "funclet"(token %root.pad) ]
    to label %unreach unwind label %right
; CHECK: [[root_cont]]:
; CHECK-NEXT: invoke void @g() [ "funclet"(token %[[root_pad]]) ]
; CHECK-NEXT:   to label %[[root_cont_cont:.+]] unwind label %cleanup
; CHECK: [[root_cont_cont]]:
; CHECK-NEXT: invoke void @g() [ "funclet"(token %[[root_pad]]) ]
; CHECK-NEXT:   to label %[[unreach]] unwind label %[[right:.+]]

right:
  %right.pad = cleanuppad within %root.pad []
  invoke void @g() [ "funclet"(token %right.pad) ]
    to label %unreach unwind label %right.child
; CHECK: [[right]]:
; CHECK-NEXT: %[[right_pad:.+]] = cleanuppad within %[[root_pad]] []
; CHECK-NEXT: invoke void @g() [ "funclet"(token %[[right_pad]]) ]
; CHECK-NEXT:   to label %[[unreach]] unwind label %[[right_child:.+]]

right.child:
  %rc.pad = cleanuppad within %right.pad []
  invoke void @g() [ "funclet"(token %rc.pad) ]
    to label %unreach unwind label %far_right
; CHECK: [[right_child]]:
; CHECK-NEXT: %[[rc_pad:.+]] = cleanuppad within %[[right_pad]] []
; CHECK-NEXT: invoke void @g() [ "funclet"(token %[[rc_pad]]) ]
; CHECK-NEXT:   to label %[[unreach]] unwind label %[[far_right:.+]]

far_right:
  %fr.cs = catchswitch within %root.pad [label %fr.catch] unwind to caller
; CHECK: [[far_right]]:
; CHECK-NEXT: %[[fr_cs:.+]] = catchswitch within %[[root_pad]] [label %[[fr_catch:.+]]] unwind label %cleanup

fr.catch:
  %fr.cp = catchpad within %fr.cs []
  unreachable
; CHECK: [[fr_catch]]:
; CHECK-NEXT: %[[fr_cp:.+]] = catchpad within %[[fr_cs]] []
; CHECK-NEXT: unreachable

unreach:
  unreachable
; CHECK: [[unreach]]:
; CHECK-NEXT: unreachable

exit:
  ret void
}


;;; Test with a no-info funclet (right) which has a cousin (left.left) that
;;; unwinds to another cousin (left.right); make sure we don't trip over this
;;; when propagating unwind destination info to "right".
;;; CHECK-LABEL: define void @test7(
define void @test7() personality void()* @ProcessCLRException {
entry:
; CHECK-NEXT: entry:
  invoke void @test7_inlinee()
    to label %exit unwind label %cleanup
cleanup:
  %pad = cleanuppad within none []
  call void @g() [ "funclet"(token %pad) ]
  cleanupret from %pad unwind to caller
exit:
  ret void
}

define void @test7_inlinee() alwaysinline personality void ()* @ProcessCLRException {
entry:
  invoke void @g()
    to label %exit unwind label %root
; CHECK-NEXT:  invoke void @g()
; CHECK-NEXT:    unwind label %[[root:.+]]

root:
  %root.cp = cleanuppad within none []
  invoke void @g() [ "funclet"(token %root.cp) ]
    to label %root.cont unwind label %child
; CHECK: [[root]]:
; CHECK-NEXT: %[[root_cp:.+]] = cleanuppad within none []
; CHECK-NEXT: invoke void @g() [ "funclet"(token %[[root_cp]]) ]
; CHECK-NEXT:   to label %[[root_cont:.+]] unwind label %[[child:.+]]

root.cont:
  cleanupret from %root.cp unwind to caller
; CHECK: [[root_cont]]:
; CHECK-NEXT: cleanupret from %[[root_cp]] unwind label %cleanup

child:
  %child.cp = cleanuppad within %root.cp []
  invoke void @g() [ "funclet"(token %child.cp) ]
    to label %child.cont unwind label %left
; CHECK: [[child]]:
; CHECK-NEXT: %[[child_cp:.+]] = cleanuppad within %[[root_cp]] []
; CHECK-NEXT: invoke void @g() [ "funclet"(token %[[child_cp]]) ]
; CHECK-NEXT:   to label %[[child_cont:.+]] unwind label %[[left:.+]]

left:
  %left.cp = cleanuppad within %child.cp []
  invoke void @g() [ "funclet"(token %left.cp) ]
    to label %left.cont unwind label %left.left
; CHECK: [[left]]:
; CHECK-NEXT: %[[left_cp:.+]] = cleanuppad within %[[child_cp]] []
; CHECK-NEXT: invoke void @g() [ "funclet"(token %[[left_cp]]) ]
; CHECK-NEXT:   to label %[[left_cont:.+]] unwind label %[[left_left:.+]]

left.left:
  %ll.cp = cleanuppad within %left.cp []
  cleanupret from %ll.cp unwind label %left.right
; CHECK: [[left_left]]:
; CHECK-NEXT: %[[ll_cp:.+]] = cleanuppad within %[[left_cp]] []
; CHECK-NEXT: cleanupret from %[[ll_cp]] unwind label %[[left_right:.+]]

left.cont:
  invoke void @g() [ "funclet"(token %left.cp) ]
    to label %unreach unwind label %left.right
; CHECK: [[left_cont]]:
; CHECK-NEXT: invoke void @g() [ "funclet"(token %[[left_cp]]) ]
; CHECK-NEXT:   to label %[[unreach:.+]] unwind label %[[left_right]]

left.right:
  %lr.cp = cleanuppad within %left.cp []
  unreachable
; CHECK: [[left_right]]:
; CHECK-NEXT: %[[lr_cp:.+]] = cleanuppad within %[[left_cp]] []
; CHECK-NEXT: unreachable

child.cont:
  invoke void @g() [ "funclet"(token %child.cp) ]
    to label %unreach unwind label %right
; CHECK: [[child_cont]]:
; CHECK-NEXT: invoke void @g() [ "funclet"(token %[[child_cp]]) ]
; CHECK-NEXT:   to label %[[unreach]] unwind label %[[right:.+]]

right:
  %right.cp = cleanuppad within %child.cp []
  call void @g() [ "funclet"(token %right.cp) ]
  unreachable
; CHECK: [[right]]:
; CHECK-NEXT: %[[right_cp:.+]] = cleanuppad within %[[child_cp]]
; CHECK-NEXT: invoke void @g() [ "funclet"(token %[[right_cp]]) ]
; CHECK-NEXT:   to label %[[right_cont:.+]] unwind label %cleanup
; CHECK: [[right_cont]]:
; CHECK-NEXT: unreachable

unreach:
  unreachable
; CHECK: [[unreach]]:
; CHECK-NEXT: unreachable

exit:
  ret void
}

declare void @ProcessCLRException()

; Make sure the logic doesn't get tripped up when the inlined invoke is
; itself within a funclet in the caller.
; CHECK-LABEL: define void @test8(
define void @test8() personality void ()* @ProcessCLRException {
entry:
  invoke void @g()
    to label %exit unwind label %callsite_parent
callsite_parent:
  %callsite_parent.pad = cleanuppad within none []
; CHECK: %callsite_parent.pad = cleanuppad within none
  invoke void @test8_inlinee() [ "funclet"(token %callsite_parent.pad) ]
    to label %ret unwind label %cleanup
ret:
  cleanupret from %callsite_parent.pad unwind label %cleanup
cleanup:
  %pad = cleanuppad within none []
  call void @g() [ "funclet"(token %pad) ]
  cleanupret from %pad unwind to caller
exit:
  ret void
}

define void @test8_inlinee() alwaysinline personality void ()* @ProcessCLRException {
entry:
  invoke void @g()
    to label %exit unwind label %inlinee_cleanup
; CHECK-NEXT: invoke void @g() [ "funclet"(token %callsite_parent.pad) ]
; CHECK-NEXT:   unwind label %[[inlinee_cleanup:.+]]

inlinee_cleanup:
  %inlinee.pad = cleanuppad within none []
  call void @g() [ "funclet"(token %inlinee.pad) ]
  unreachable
; CHECK: [[inlinee_cleanup]]:
; CHECK-NEXT: %[[inlinee_pad:[^ ]+]] = cleanuppad within %callsite_parent.pad
; CHECK-NEXT: invoke void @g() [ "funclet"(token %[[inlinee_pad]]) ]
; CHECK-NEXT:   unwind label %cleanup{{$}}

exit:
  ret void
}
