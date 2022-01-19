; RUN: opt -S -passes=deadargelim < %s | FileCheck %s
target triple = "x86_64-pc-windows-msvc"

define internal void @callee(i8*) {
entry:
  call void @thunk()
  ret void
}

define void @test1() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @thunk()
          to label %good1 unwind label %bad1

good1:                                            ; preds = %entry-block
  ret void

bad1:                                             ; preds = %entry-block
  %pad1 = cleanuppad within none []
  call void @callee(i8* null) [ "funclet"(token %pad1) ]
  cleanupret from %pad1 unwind to caller
}
; CHECK-LABEL: define void @test1(
; CHECK:      %[[pad:.*]] = cleanuppad within none []
; CHECK-NEXT: call void @callee() [ "funclet"(token %[[pad]]) ]

declare void @thunk()

declare i32 @__CxxFrameHandler3(...)
