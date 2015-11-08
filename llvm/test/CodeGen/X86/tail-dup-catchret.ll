; RUN: llc < %s | FileCheck %s
target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc18.0.0"

define void @f() personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  invoke void @g()
          to label %try.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %0 = catchpad [i8* null, i32 64, i8* null]
          to label %catch unwind label %catchendblock

catch:                                            ; preds = %catch.dispatch
  catchret %0 to label %try.cont

try.cont:                                         ; preds = %entry, %catch
  %b.0 = phi i1 [ false, %catch ], [ true, %entry ]
  tail call void @h(i1 zeroext %b.0)
  ret void

catchendblock:                                    ; preds = %catch.dispatch
  catchendpad unwind to caller
}

; CHECK-LABEL: _f:
; CHECK: calll _g
; CHECK: calll _h

declare void @g()

declare i32 @__CxxFrameHandler3(...)

declare void @h(i1 zeroext)
