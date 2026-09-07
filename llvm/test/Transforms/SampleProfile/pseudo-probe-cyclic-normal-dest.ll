; RUN: opt < %s -passes=pseudo-probe -S | FileCheck %s

; getOriginalTerminator walks invoke normal destinations. If that chain is a
; cycle (self-looping invoke), the walk must stop. This CFG hangs without
; cycle detection.
;
; Matches clang++ -O2 -fexceptions on: try { for (;;) f1(); } catch (...) {}
define void @self_looping_invoke() personality ptr @__gxx_personality_v0 {
; CHECK-LABEL: define void @self_looping_invoke(
entry:
; CHECK:      entry:
; CHECK:        call void @llvm.pseudoprobe
; CHECK-NEXT:   br label %loop
  br label %loop

loop:
; CHECK:      loop:
; CHECK-NEXT:   invoke void @may_throw()
; CHECK-NEXT:           to label %loop unwind label %lpad
  invoke void @may_throw()
          to label %loop unwind label %lpad

lpad:
; CHECK:      lpad:
; CHECK-NOT:    call void @llvm.pseudoprobe
  %eh = landingpad { ptr, i32 }
          cleanup
  ret void
}

declare void @may_throw()
declare i32 @__gxx_personality_v0(...)
