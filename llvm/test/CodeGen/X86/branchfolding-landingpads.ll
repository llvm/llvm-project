; RUN: llc %s -o - -verify-machineinstrs | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

; The machine level BranchFolding pass will try to remove the 'unreachable' block
; and rewrite 'entry' to jump to the block 'unreachable' falls through to.
; That will be a landing pad and result in 'entry' jumping to 2 landing pads.
; This tests that we don't do this change when the fallthrough is itself a landing
; pad.

declare i32 @__gxx_personality_v0(...)
declare void @foo()

; Function Attrs: noreturn
declare void @_throw()

; CHECK-LABEL: @main
; CHECK: %unreachable

define i32 @main(ptr %cleanup) personality ptr @__gxx_personality_v0 {
entry:
  invoke void @_throw() #0
          to label %unreachable unwind label %catch.dispatch9

catch.dispatch9:                                  ; preds = %entry
  %tmp13 = landingpad { ptr, i32 }
          cleanup
          catch ptr null
  invoke void @_throw() #0
          to label %unreachable unwind label %lpad31

lpad31:                                           ; preds = %catch.dispatch9
  %tmp20 = landingpad { ptr, i32 }
          cleanup
          catch ptr null
  call void @foo()
  unreachable

unreachable:                                      ; preds = %catch.dispatch9, %entry
  unreachable
}

attributes #0 = { noreturn }

