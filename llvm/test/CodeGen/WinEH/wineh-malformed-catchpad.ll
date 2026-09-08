; RUN: split-file %s %t
; RUN: llc -o - %t/issue219223_cxx.ll | FileCheck %t/issue219223_cxx.ll
; RUN: llc -o - %t/issue219223_seh.ll | FileCheck %t/issue219223_seh.ll
; RUN: llc -o - %t/issue219223_clr.ll | FileCheck %t/issue219223_clr.ll
; RUN: opt -mtriple=x86_64-pc-windows-msvc -S -passes=win-eh-prepare < %t/single.ll | FileCheck %t/single.ll
; RUN: opt -mtriple=x86_64-pc-windows-msvc -S -passes=win-eh-prepare < %t/sibling_cxx.ll | FileCheck %t/sibling_cxx.ll
; RUN: opt -mtriple=x86_64-pc-windows-msvc -S -passes=win-eh-prepare < %t/sibling_seh.ll | FileCheck %t/sibling_seh.ll
; RUN: opt -mtriple=x86_64-pc-windows-msvc -S -passes=win-eh-prepare < %t/sibling_clr.ll | FileCheck %t/sibling_clr.ll

;--- issue219223_cxx.ll
target triple = "x86_64-pc-linux-gnu"

declare i32 @__CxxFrameHandler3(...)
declare void @f()

define void @empty_catchpad_cxx() personality ptr @__CxxFrameHandler3 {
entry:
  invoke void @f() to label %cont unwind label %dispatch

dispatch:
  %cs = catchswitch within none [label %catch] unwind to caller

catch:
  %pad = catchpad within %cs []
  catchret from %pad to label %cont

cont:
  ret void
}

; CHECK-LABEL: empty_catchpad_cxx:

;--- issue219223_seh.ll
declare i32 @__C_specific_handler(...)
declare void @f()

define void @empty_catchpad_seh() personality ptr @__C_specific_handler {
entry:
  invoke void @f() to label %cont unwind label %dispatch

dispatch:
  %cs = catchswitch within none [label %catch] unwind to caller

catch:
  %pad = catchpad within %cs []
  catchret from %pad to label %cont

cont:
  ret void
}

; CHECK-LABEL: empty_catchpad_seh:

;--- issue219223_clr.ll
declare void @ProcessCLRException(...)
declare void @f()

define void @empty_catchpad_clr() personality ptr @ProcessCLRException {
entry:
  invoke void @f() to label %cont unwind label %dispatch

dispatch:
  %cs = catchswitch within none [label %catch] unwind to caller

catch:
  %pad = catchpad within %cs []
  catchret from %pad to label %cont

cont:
  ret void
}

; CHECK-LABEL: empty_catchpad_clr:

;--- single.ll
declare i32 @__CxxFrameHandler3(...)
declare void @f()

define void @malformed_single() personality ptr @__CxxFrameHandler3 {
entry:
  invoke void @f()
          to label %cont unwind label %dispatch

dispatch:
  %cs = catchswitch within none [label %catch] unwind to caller

catch:
  %pad = catchpad within %cs []
  catchret from %pad to label %cont

cont:
  ret void
}

; CHECK-LABEL: define void @malformed_single()
; CHECK:      %cs = catchswitch within none [label %catch] unwind to caller
; CHECK:      catch:
; CHECK-NEXT:   %pad = catchpad within %cs []
; CHECK-NEXT:   unreachable
; CHECK: cont: ; preds = %entry
; CHECK-NEXT: ret void

;--- sibling_cxx.ll
declare i32 @__CxxFrameHandler3(...)
declare void @f()

define void @sibling_cxx() personality ptr @__CxxFrameHandler3 {
entry:
  invoke void @f()
          to label %cont unwind label %dispatch

dispatch:
  %cs = catchswitch within none [label %catch.bad, label %catch.good] unwind to caller

catch.bad:
  %pad.bad = catchpad within %cs [ptr null]
  catchret from %pad.bad to label %cont

catch.good:
  %pad.good = catchpad within %cs [ptr null, i32 0, ptr null]
  catchret from %pad.good to label %cont

cont:
  ret void
}

; CHECK-LABEL: define void @sibling_cxx()
; CHECK:      catch.bad:
; CHECK-NEXT:   %pad.bad = catchpad within %cs [ptr null]
; CHECK-NEXT:   unreachable
; CHECK:      catch.good:
; CHECK-NEXT:   %pad.good = catchpad within %cs [ptr null, i32 0, ptr null]
; CHECK-NEXT:   unreachable
; CHECK: cont: ; preds = %entry
; CHECK-NEXT: ret void

;--- sibling_seh.ll
declare i32 @__C_specific_handler(...)
declare void @f()

define void @sibling_seh() personality ptr @__C_specific_handler {
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

; CHECK-LABEL: define void @sibling_seh()
; CHECK:      catch.bad:
; CHECK-NEXT:   %pad.bad = catchpad within %cs []
; CHECK-NEXT:   unreachable
; CHECK:      catch.good:
; CHECK-NEXT:   %pad.good = catchpad within %cs [ptr null]
; CHECK-NEXT:   unreachable
; CHECK: cont: ; preds = %entry
; CHECK-NEXT: ret void

;--- sibling_clr.ll
declare void @ProcessCLRException(...)
declare void @f()

define void @sibling_clr() personality ptr @ProcessCLRException {
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

; CHECK-LABEL: define void @sibling_clr()
; CHECK:      catch.bad:
; CHECK-NEXT:   %pad.bad = catchpad within %cs []
; CHECK-NEXT:   unreachable
; CHECK:      catch.good:
; CHECK-NEXT:   %pad.good = catchpad within %cs [i32 1]
; CHECK-NEXT:   unreachable
; CHECK: cont: ; preds = %entry
; CHECK-NEXT: ret void
