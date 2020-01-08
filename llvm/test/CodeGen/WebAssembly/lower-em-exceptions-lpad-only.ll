; RUN: opt < %s -wasm-lower-em-ehsjlj -S | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

@_ZTIi = external constant i8*

; Checks if a module that only contains a landingpad (and resume) but not an
; invoke works correctly and does not crash.
; CHECK-LABEL: @landingpad_only
define void @landingpad_only() personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  br label %cont

lpad:                                             ; preds = %entry
  %0 = landingpad { i8*, i32 }
          catch i8* bitcast (i8** @_ZTIi to i8*)
          catch i8* null
  resume { i8*, i32 } %0

cont:
  ret void
}

declare i32 @__gxx_personality_v0(...)
