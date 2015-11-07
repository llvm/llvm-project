; RUN: opt < %s -instcombine -S | FileCheck %s
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc18.0.0"

declare i32 @__CxxFrameHandler3(...)

define void @test1() personality i32 (...)* @__CxxFrameHandler3 {
bb:
  unreachable

unreachable:
  %cl = cleanuppad []
  cleanupret %cl unwind to caller
}

; CHECK-LABEL: define void @test1(
; CHECK: unreachable:
; CHECK:   %cl = cleanuppad []
; CHECK:   cleanupret %cl unwind to caller

define void @test2(i8 %A, i8 %B) personality i32 (...)* @__CxxFrameHandler3 {
bb:
  %X = zext i8 %A to i32
  invoke void @g(i32 0)
    to label %cont
    unwind label %catch

cont:
  %Y = zext i8 %B to i32
  invoke void @g(i32 0)
    to label %unreachable
    unwind label %catch

catch:
  %phi = phi i32 [ %X, %bb ], [ %Y, %cont ]
  %cl = catchpad []
   to label %doit
   unwind label %endpad

doit:
  call void @g(i32 %phi)
  unreachable

unreachable:
  unreachable

endpad:
  catchendpad unwind to caller
}

; CHECK-LABEL: define void @test2(
; CHECK:  %X = zext i8 %A to i32
; CHECK:  %Y = zext i8 %B to i32
; CHECK:  %phi = phi i32 [ %X, %bb ], [ %Y, %cont ]

define void @test3(i8 %A, i8 %B) personality i32 (...)* @__CxxFrameHandler3 {
bb:
  %X = zext i8 %A to i32
  invoke void @g(i32 0)
    to label %cont
    unwind label %catch

cont:
  %Y = zext i8 %B to i32
  invoke void @g(i32 0)
    to label %cont2
    unwind label %catch

cont2:
  invoke void @g(i32 0)
    to label %unreachable
    unwind label %catch

catch:
  %phi = phi i32 [ %X, %bb ], [ %Y, %cont ], [ %Y, %cont2 ]
  %cl = catchpad []
   to label %doit
   unwind label %endpad

doit:
  call void @g(i32 %phi)
  unreachable

unreachable:
  unreachable

endpad:
  catchendpad unwind to caller
}

; CHECK-LABEL: define void @test3(
; CHECK:  %X = zext i8 %A to i32
; CHECK:  %Y = zext i8 %B to i32
; CHECK:  %phi = phi i32 [ %X, %bb ], [ %Y, %cont ], [ %Y, %cont2 ]


declare void @g(i32)
