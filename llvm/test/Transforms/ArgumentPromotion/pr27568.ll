; RUN: opt -S -argpromotion < %s | FileCheck %s
target triple = "x86_64-pc-windows-msvc"

define internal void @callee(i8*) {
entry:
  call void @thunk()
  ret void
}

define void @test1() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @thunk()
          to label %out unwind label %cpad

out:
  ret void

cpad:
  %pad = cleanuppad within none []
  call void @callee(i8* null) [ "funclet"(token %pad) ]
  cleanupret from %pad unwind to caller
}

; CHECK-LABEL: define void @test1(
; CHECK:      %[[pad:.*]] = cleanuppad within none []
; CHECK-NEXT: call void @callee() [ "funclet"(token %[[pad]]) ]
; CHECK-NEXT: cleanupret from %[[pad]] unwind to caller

declare void @thunk()

declare i32 @__CxxFrameHandler3(...)
