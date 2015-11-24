; RUN: opt -codegenprepare -S < %s | FileCheck %s

; The following target lines are needed for the test to exercise what it should.
; Without these lines, CodeGenPrepare does not try to sink the bitcasts.
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

declare i32 @__CxxFrameHandler3(...)

declare void @f()

declare void @g(i8*)

; CodeGenPrepare will want to sink these bitcasts, but it selects the catchpad
; blocks as the place to which the bitcast should be sunk.  Since catchpads
; do not allow non-phi instructions before the terminator, this isn't possible. 

; CHECK-LABEL: @test(
define void @test(i32* %addr) personality i32 (...)* @__CxxFrameHandler3 {
; CHECK: entry:
; CHECK-NEXT: %x = getelementptr i32, i32* %addr, i32 1
; CHECK-NEXT: %p1 = bitcast i32* %x to i8*
entry:
  %x = getelementptr i32, i32* %addr, i32 1
  %p1 = bitcast i32* %x to i8*
  invoke void @f()
          to label %invoke.cont unwind label %catch1

; CHECK: invoke.cont:
; CHECK-NEXT: %y = getelementptr i32, i32* %addr, i32 2
; CHECK-NEXT: %p2 = bitcast i32* %y to i8*
invoke.cont:
  %y = getelementptr i32, i32* %addr, i32 2
  %p2 = bitcast i32* %y to i8*
  invoke void @f()
          to label %done unwind label %catch2

done:
  ret void

catch1:
  %cp1 = catchpad [] to label %catch.dispatch unwind label %catchend1

catch2:
  %cp2 = catchpad [] to label %catch.dispatch unwind label %catchend2

; CHECK: catch.dispatch:
; CHECK-NEXT: %p = phi i8* [ %p1, %catch1 ], [ %p2, %catch2 ]
catch.dispatch:
  %p = phi i8* [ %p1, %catch1 ], [ %p2, %catch2 ]
  call void @g(i8* %p)
  unreachable

catchend1:
  catchendpad unwind to caller

catchend2:
  catchendpad unwind to caller
}
