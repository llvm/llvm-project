; RUN: llc < %s | FileCheck %s
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc18.0.0"

; Function Attrs: uwtable
define void @f() #0 personality ptr @__CxxFrameHandler3 {
entry:
  invoke void @g()
          to label %try.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %0 = catchswitch within none [label %catch] unwind label %ehcleanup

catch:                                            ; preds = %catch.dispatch
  %1 = catchpad within %0 [ptr null, i32 64, ptr null]
  invoke void @g() [ "funclet"(token %1) ]
          to label %dtor.exit unwind label %catch.dispatch.i

catch.dispatch.i:                                 ; preds = %catch
  %2 = catchswitch within %1 [label %catch.i] unwind to caller

catch.i:                                          ; preds = %catch.dispatch.i
  %3 = catchpad within %2 [ptr null, i32 64, ptr null]
  catchret from %3 to label %dtor.exit

dtor.exit:
  catchret from %1 to label %try.cont

try.cont:
  ret void

ehcleanup:                                        ; preds = %catch.dispatch
  %4 = cleanuppad within none []
  call void @dtor() #1 [ "funclet"(token %4) ]
  cleanupret from %4 unwind to caller
}

declare void @g()

declare i32 @__CxxFrameHandler3(...)

; Function Attrs: nounwind
declare void @dtor() #1

attributes #0 = { uwtable }
attributes #1 = { nounwind }

; CHECK-LABEL: $ip2state$f:
; CHECK: -1
; CHECK: 1
; CHECK: -1
; CHECK: 4
; CHECK: 2
; CHECK: 3
; CHECK: 2
