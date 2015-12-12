; RUN: opt < %s -inline -S | FileCheck %s
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc18.0.0"

define void @f() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @g()
          to label %try.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %cs1 = catchswitch within none [label %catch] unwind to caller

catch:                                            ; preds = %catch.dispatch
  %0 = catchpad within %cs1 [i8* null, i32 64, i8* null]
  invoke void @dtor()
          to label %invoke.cont.1 unwind label %ehcleanup

invoke.cont.1:                                    ; preds = %catch
  catchret from %0 to label %try.cont

try.cont:                                         ; preds = %entry, %invoke.cont.1
  ret void

ehcleanup:
  %cp2 = cleanuppad within none []
  call void @g()
  cleanupret from %cp2 unwind to caller
}

; CHECK-LABEL:  define void @f(

; CHECK:         invoke void @g()
; CHECK:                 to label %dtor.exit unwind label %terminate.i

; CHECK:       terminate.i:
; CHECK-NEXT:    terminatepad within %0 [void ()* @terminate] unwind label %ehcleanup

declare i32 @__CxxFrameHandler3(...)

define internal void @dtor() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @g()
          to label %invoke.cont unwind label %terminate

invoke.cont:                                      ; preds = %entry
  ret void

terminate:                                        ; preds = %entry
  terminatepad within none [void ()* @terminate] unwind to caller
}

declare void @g()
declare void @terminate()
