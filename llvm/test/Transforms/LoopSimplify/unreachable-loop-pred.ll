; RUN: opt -S -passes=loop-simplify -verify-loop-info -verify-dom-info < %s | FileCheck %s
; PR5235

; When loopsimplify inserts a preheader for this loop, it should add the new
; block to the enclosing loop and not get confused by the unreachable
; bogus loop entry.

define void @is_extract_cab(i1 %arg) nounwind {
entry:
  br label %header

header:                                       ; preds = %if.end206, %cond.end66, %if.end23
  br label %while.body115

while.body115:                                    ; preds = %9, %if.end192, %if.end101
  br i1 %arg, label %header, label %while.body115

foo:
  br label %while.body115
}

; When loopsimplify generates dedicated exit block for blocks that are landing
; pads (i.e. innerLoopExit in this test), we should not get confused with the
; unreachable pred (unreachableB) to innerLoopExit.
define void @baz(i32 %trip) personality ptr @wobble {
entry:
  br label %outerHeader

outerHeader:
  invoke void @foo()
          to label %innerPreheader unwind label %innerLoopExit

innerPreheader:
  br label %innerH

innerH:
  %tmp50 = invoke ptr undef()
          to label %innerLatch unwind label %innerLoopExit

innerLatch:
  %cmp = icmp slt i32 %trip, 42
  br i1 %cmp, label %innerH, label %retblock

unreachableB:                                             ; No predecessors!
  %tmp62 = invoke ptr undef()
          to label %retblock unwind label %innerLoopExit

; undedicated exit block (preds from inner and outer loop)
; Also has unreachableB as pred.
innerLoopExit:
  %tmp65 = landingpad { ptr, i32 }
          cleanup
  invoke void @foo()
          to label %outerHeader unwind label %unwindblock

unwindblock:
  %tmp67 = landingpad { ptr, i32 }
          cleanup
  ret void

retblock:
  ret void
}

; Ensure that removing >= 5 unreachable out-of-loop predecessors preserves a
; deterministic incoming block order in the remaining PHINode operands.
define i32 @deterministic_dead_pred_removal(i32 %x, i1 %cond) {
; CHECK-LABEL: define i32 @deterministic_dead_pred_removal(
; CHECK:       latch:
; CHECK-NEXT:    %v = phi i32 [ 10, %live1 ], [ 20, %live2 ], [ 30, %live3 ], [ 40, %live4 ], [ 50, %live5 ]
; CHECK-NEXT:    br i1 %cond, label %header, label %exit
entry:
  br label %header

header:
  switch i32 %x, label %live1 [
    i32 1, label %live2
    i32 2, label %live3
    i32 3, label %live4
    i32 4, label %live5
  ]

live1:
  br label %latch

live2:
  br label %latch

live3:
  br label %latch

live4:
  br label %latch

live5:
  br label %latch

dead1:
  br label %latch

dead2:
  br label %latch

dead3:
  br label %latch

dead4:
  br label %latch

dead5:
  br label %latch

latch:
  %v = phi i32 [ 1, %dead1 ], [ 2, %dead2 ], [ 3, %dead3 ], [ 4, %dead4 ], [ 5, %dead5 ], [ 10, %live1 ], [ 20, %live2 ], [ 30, %live3 ], [ 40, %live4 ], [ 50, %live5 ]
  br i1 %cond, label %header, label %exit

exit:
  ret i32 %v
}

; Function Attrs: nounwind
declare ptr @wobble()

; Function Attrs: uwtable
declare void @foo()
