; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-block-placement -wasm-disable-explicit-locals -wasm-keep-registers -enable-emscripten-cxx-exceptions | FileCheck %s

target triple = "wasm32-unknown-unknown"

declare i32 @__gxx_personality_v0(...)

; Check an interesting case of complex control flow due to exceptions CFG rewriting.
; There should *not* be any irreducible control flow here.

; CHECK-LABEL: crashy:
; CHECK-NOT: br_table

define void @crashy() personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  invoke void undef()
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  invoke void undef()
          to label %invoke.cont4 unwind label %lpad3

invoke.cont4:                                     ; preds = %invoke.cont
  %call.i82 = invoke i8* undef()
          to label %invoke.cont6 unwind label %lpad3

invoke.cont6:                                     ; preds = %invoke.cont4
  invoke void undef()
          to label %invoke.cont13 unwind label %lpad12

invoke.cont13:                                    ; preds = %invoke.cont6
  br label %for.cond

for.cond:                                         ; preds = %for.cond.backedge, %invoke.cont13
  br i1 undef, label %exit2, label %land.lhs

land.lhs:                                         ; preds = %for.cond
  %call.i.i.i.i92 = invoke i32 undef()
          to label %exit1 unwind label %lpad16.loopexit

exit1:                                            ; preds = %land.lhs
  br label %exit2

exit2:                                            ; preds = %exit1, %for.cond
  %call.i.i12.i.i93 = invoke i32 undef()
          to label %exit3 unwind label %lpad16.loopexit

exit3:                                            ; preds = %exit2
  invoke void undef()
          to label %invoke.cont23 unwind label %lpad22

invoke.cont23:                                    ; preds = %exit3
  invoke void undef()
          to label %invoke.cont25 unwind label %lpad22

invoke.cont25:                                    ; preds = %invoke.cont23
  %call.i.i137 = invoke i32 undef()
          to label %invoke.cont29 unwind label %lpad16.loopexit

lpad:                                             ; preds = %entry
  %0 = landingpad { i8*, i32 }
          cleanup
  unreachable

lpad3:                                            ; preds = %invoke.cont4, %invoke.cont
  %1 = landingpad { i8*, i32 }
          cleanup
  unreachable

lpad12:                                           ; preds = %invoke.cont6
  %2 = landingpad { i8*, i32 }
          cleanup
  resume { i8*, i32 } undef

lpad16.loopexit:                                  ; preds = %if.then, %invoke.cont29, %invoke.cont25, %exit2, %land.lhs
  %lpad.loopexit = landingpad { i8*, i32 }
          cleanup
  unreachable

lpad22:                                           ; preds = %invoke.cont23, %exit3
  %3 = landingpad { i8*, i32 }
          cleanup
  unreachable

invoke.cont29:                                    ; preds = %invoke.cont25
  invoke void undef()
          to label %invoke.cont33 unwind label %lpad16.loopexit

invoke.cont33:                                    ; preds = %invoke.cont29
  br label %for.inc

for.inc:                                          ; preds = %invoke.cont33
  %cmp.i.i141 = icmp eq i8* undef, undef
  br i1 %cmp.i.i141, label %if.then, label %if.end.i.i146

if.then:                                          ; preds = %for.inc
  %call.i.i148 = invoke i32 undef()
          to label %for.cond.backedge unwind label %lpad16.loopexit

for.cond.backedge:                                ; preds = %if.end.i.i146, %if.then
  br label %for.cond

if.end.i.i146:                                    ; preds = %for.inc
  call void undef()
  br label %for.cond.backedge
}
