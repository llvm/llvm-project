; RUN: opt < %s -inline -S | FileCheck %s
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc18.0.0"

define void @f() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @g()
          to label %try.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %0 = catchpad [i8* null, i32 64, i8* null]
          to label %catch unwind label %catchendblock

catch:                                            ; preds = %catch.dispatch
  invoke void @dtor()
          to label %invoke.cont.1 unwind label %catchendblock

invoke.cont.1:                                    ; preds = %catch
  catchret %0 to label %try.cont

try.cont:                                         ; preds = %entry, %invoke.cont.1
  ret void

catchendblock:                                    ; preds = %catch, %catch.dispatch
  catchendpad unwind to caller
}

; CHECK-LABEL:  define void @f(

; CHECK:         invoke void @g()
; CHECK:                 to label %dtor.exit unwind label %terminate.i

; CHECK:       terminate.i:
; CHECK-NEXT:    terminatepad [void ()* @terminate] unwind label %catchendblock

; CHECK:       catchendblock:
; CHECK-NEXT:    catchendpad unwind to caller

declare i32 @__CxxFrameHandler3(...)

define internal void @dtor() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @g()
          to label %invoke.cont unwind label %terminate

invoke.cont:                                      ; preds = %entry
  ret void

terminate:                                        ; preds = %entry
  terminatepad [void ()* @terminate] unwind to caller
}

declare void @g()
declare void @terminate()
