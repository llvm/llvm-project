; RUN: split-file %s %t
; RUN: not opt -mtriple=x86_64-pc-windows-msvc -S -passes=win-eh-prepare < %t/cxx.ll 2>/dev/null | FileCheck %t/cxx.ll
; RUN: not opt -mtriple=x86_64-pc-windows-msvc -S -passes=win-eh-prepare < %t/seh.ll 2>/dev/null | FileCheck %t/seh.ll
; RUN: not opt -mtriple=x86_64-pc-windows-msvc -S -passes=win-eh-prepare < %t/clr.ll 2>/dev/null | FileCheck %t/clr.ll

;--- cxx.ll
declare i32 @__CxxFrameHandler3(...)
declare void @f()

define void @cxx() personality ptr @__CxxFrameHandler3 {
entry:
  invoke void @f()
  to label %cont unwind label %dispatch

dispatch:
  %cs = catchswitch within none [label %catch.bad, label %catch.good] unwind to caller

catch.bad:
  %pad.bad = catchpad within %cs [ptr null]
  catchret from %pad.bad to label %cont

catch.good:
  %pad.good = catchpad within %cs [ptr null, i32 1, ptr null]
  catchret from %pad.good to label %cont

cont:
  ret void
}

; CHECK-LABEL: define void @cxx()
; CHECK:      catch.bad:
; CHECK-NEXT:   %pad.bad1 = catchpad within %cs [ptr null, i32 0, ptr null]
; CHECK-NEXT:   unreachable
; CHECK: cont: ; preds = %entry
; CHECK-NEXT: ret void

;--- seh.ll
declare i32 @__C_specific_handler(...)
declare void @f()

define void @seh() personality ptr @__C_specific_handler {
entry:
  invoke void @f()
  to label %cont unwind label %dispatch

dispatch:
  %cs = catchswitch within none [label %catch.bad, label %catch.good] unwind to caller

catch.bad:
  %pad.bad = catchpad within %cs []
  catchret from %pad.bad to label %cont

catch.good:
  %pad.good = catchpad within %cs [ptr null]
  catchret from %pad.good to label %cont

cont:
  ret void
}

; CHECK-LABEL: define void @seh()
; CHECK:      catch.bad:
; CHECK-NEXT:   %pad.bad1 = catchpad within %cs [ptr null]
; CHECK-NEXT:   unreachable
; CHECK: cont: ; preds = %entry
; CHECK-NEXT: ret void

;--- clr.ll
declare void @ProcessCLRException(...)
declare void @f()

define void @clr() personality ptr @ProcessCLRException {
entry:
  invoke void @f()
  to label %cont unwind label %dispatch

dispatch:
  %cs = catchswitch within none [label %catch.bad, label %catch.good] unwind to caller

catch.bad:
  %pad.bad = catchpad within %cs []
  catchret from %pad.bad to label %cont

catch.good:
  %pad.good = catchpad within %cs [i32 1]
  catchret from %pad.good to label %cont

cont:
  ret void
}

; CHECK-LABEL: define void @clr()
; CHECK:      catch.bad:
; CHECK-NEXT:   %pad.bad1 = catchpad within %cs [i32 0]
; CHECK-NEXT:   unreachable
; CHECK: cont: ; preds = %entry
; CHECK-NEXT: ret void
