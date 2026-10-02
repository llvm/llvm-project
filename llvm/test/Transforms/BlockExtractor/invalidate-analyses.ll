; RUN: opt -passes='function(require<domtree>),extract-blocks,function(verify<domtree>)' -disable-output %s

; The landing pad split runs even with an empty block list, so the CFG changes.
; Ensure that we properly invalidate analyses.

define void @foo() personality ptr @__gxx_personality_v0 {
entry:
  invoke void @bar()
          to label %exit unwind label %lpad

lpad:
  %0 = landingpad { ptr, i32 }
          catch ptr null
  invoke void @bar()
          to label %exit unwind label %lpad2

lpad2:
  %1 = landingpad { ptr, i32 }
          cleanup
  invoke void @bar()
          to label %exit unwind label %lpad

exit:
  ret void
}

declare void @bar()

declare i32 @__gxx_personality_v0(...)
