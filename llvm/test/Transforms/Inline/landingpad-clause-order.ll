; Inlining through an invoke appends the outer landing pad's clauses to every
; inlined landing pad. The landing pads must be visited in block order, not
; pointer order, so the use-list order of @ti is the same in every run.

; RUN: opt -passes=inline -S -preserve-ll-uselistorder %s | FileCheck %s

; CHECK: uselistorder ptr @ti, { 17, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 }

@ti = external constant ptr

declare void @may_throw()
declare i32 @__gxx_personality_v0(...)

define void @outer() personality ptr @__gxx_personality_v0 {
  invoke void @inner() to label %ok unwind label %lp
ok:
  ret void
lp:
  %l = landingpad { ptr, i32 } catch ptr @ti
  ret void
}

; Use more than 16 landing pads. A SmallPtrSet<_, 16> would then iterate them
; in pointer order.
define internal void @inner() personality ptr @__gxx_personality_v0 {
  invoke void @may_throw() to label %b1 unwind label %lp0
b1:
  invoke void @may_throw() to label %b2 unwind label %lp1
b2:
  invoke void @may_throw() to label %b3 unwind label %lp2
b3:
  invoke void @may_throw() to label %b4 unwind label %lp3
b4:
  invoke void @may_throw() to label %b5 unwind label %lp4
b5:
  invoke void @may_throw() to label %b6 unwind label %lp5
b6:
  invoke void @may_throw() to label %b7 unwind label %lp6
b7:
  invoke void @may_throw() to label %b8 unwind label %lp7
b8:
  invoke void @may_throw() to label %b9 unwind label %lp8
b9:
  invoke void @may_throw() to label %b10 unwind label %lp9
b10:
  invoke void @may_throw() to label %b11 unwind label %lp10
b11:
  invoke void @may_throw() to label %b12 unwind label %lp11
b12:
  invoke void @may_throw() to label %b13 unwind label %lp12
b13:
  invoke void @may_throw() to label %b14 unwind label %lp13
b14:
  invoke void @may_throw() to label %b15 unwind label %lp14
b15:
  invoke void @may_throw() to label %b16 unwind label %lp15
b16:
  invoke void @may_throw() to label %done unwind label %lp16
lp0:
  %l0 = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } %l0
lp1:
  %l1 = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } %l1
lp2:
  %l2 = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } %l2
lp3:
  %l3 = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } %l3
lp4:
  %l4 = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } %l4
lp5:
  %l5 = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } %l5
lp6:
  %l6 = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } %l6
lp7:
  %l7 = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } %l7
lp8:
  %l8 = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } %l8
lp9:
  %l9 = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } %l9
lp10:
  %l10 = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } %l10
lp11:
  %l11 = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } %l11
lp12:
  %l12 = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } %l12
lp13:
  %l13 = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } %l13
lp14:
  %l14 = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } %l14
lp15:
  %l15 = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } %l15
lp16:
  %l16 = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } %l16
done:
  ret void
}
