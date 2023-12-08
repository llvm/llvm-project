; RUN: opt -S -passes=loop-simplify -disable-output -verify-loop-info -verify-dom-info < %s
; PR5235

; When loopsimplify inserts a preheader for this loop, it should add the new
; block to the enclosing loop and not get confused by the unreachable
; bogus loop entry.

define void @is_extract_cab() nounwind {
entry:
  br label %header

header:                                       ; preds = %if.end206, %cond.end66, %if.end23
  br label %while.body115

while.body115:                                    ; preds = %9, %if.end192, %if.end101
  br i1 undef, label %header, label %while.body115

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

; Function Attrs: nounwind
declare ptr @wobble()

; Function Attrs: uwtable
declare void @foo()
