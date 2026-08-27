; RUN: opt -p loop-unroll -unroll-full-max-count=0 -S --preserve-ll-uselistorder < %s | FileCheck %s

; Peeling the last iteration replaces each LCSSA phi in the exit block with the
; corresponding value from the peeled iteration. When several phis share an
; incoming value, the order in which they are replaced becomes the use list
; order of that value, so it must not depend on the iteration order of a
; pointer-keyed map.
;
; The number of phis matters. DenseMapInfo<T *> hashes (ptr >> 4) ^ (ptr >> 9),
; so with only a few phis the hash order coincides with the exit block order
; often enough for the test to pass even without the fix: four phis failed 35 of
; 50 runs of an ordinary build and six failed 46 of 50. The eight used here
; failed every run.

define void @peel_last_shared_exit_value(i32 %n) {
; CHECK-LABEL: define void @peel_last_shared_exit_value(
; CHECK: %sel.peel = select i1 %c.peel, i32 1, i32 2
; CHECK-NOT: uselistorder i32 %sel.peel
entry:
  %sub = add i32 %n, -1
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %c = icmp eq i32 %iv, %sub
  %sel = select i1 %c, i32 1, i32 2
  call void @foo(i32 %sel)
  %iv.next = add i32 %iv, 1
  %ec = icmp ne i32 %iv.next, %n
  br i1 %ec, label %loop, label %exit

exit:
  %p0 = phi i32 [ %sel, %loop ]
  %p1 = phi i32 [ %sel, %loop ]
  %p2 = phi i32 [ %sel, %loop ]
  %p3 = phi i32 [ %sel, %loop ]
  %p4 = phi i32 [ %sel, %loop ]
  %p5 = phi i32 [ %sel, %loop ]
  %p6 = phi i32 [ %sel, %loop ]
  %p7 = phi i32 [ %sel, %loop ]
  call void @use(i32 %p0)
  call void @use(i32 %p1)
  call void @use(i32 %p2)
  call void @use(i32 %p3)
  call void @use(i32 %p4)
  call void @use(i32 %p5)
  call void @use(i32 %p6)
  call void @use(i32 %p7)
  ret void
}

declare void @foo(i32)
declare void @use(i32)
