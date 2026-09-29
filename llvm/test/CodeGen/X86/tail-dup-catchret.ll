; RUN: llc < %s | FileCheck %s
target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc18.0.0"

define void @f() personality ptr @__CxxFrameHandler3 {
entry:
  invoke void @g()
          to label %try.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %cs1 = catchswitch within none [label %catch] unwind to caller

catch:                                            ; preds = %catch.dispatch
  %0 = catchpad within %cs1 [ptr null, i32 64, ptr null]
  catchret from %0 to label %try.cont

try.cont:                                         ; preds = %entry, %catch
  %b.0 = phi i1 [ false, %catch ], [ true, %entry ]
  tail call void @h(i1 zeroext %b.0)
  ret void
}

; CHECK-LABEL: _f:
; CHECK: calll _g
; CHECK: calll _h

declare void @g()

declare i32 @__CxxFrameHandler3(...)

declare void @h(i1 zeroext)
