; RUN: opt < %s -passes=instcombine -S | FileCheck %s
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc18.0.0"

declare i32 @__CxxFrameHandler3(...)

define void @test1() personality i32 (...)* @__CxxFrameHandler3 {
bb:
  unreachable

unreachable:
  %cl = cleanuppad within none []
  cleanupret from %cl unwind to caller
}

; CHECK-LABEL: define void @test1(
; CHECK: unreachable:
; CHECK:   %cl = cleanuppad within none []
; CHECK:   cleanupret from %cl unwind to caller

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
  %cs = catchswitch within none [label %doit] unwind to caller

doit:
  %cl = catchpad within %cs []
  call void @g(i32 %phi)
  unreachable

unreachable:
  unreachable
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
  %cs = catchswitch within none [label %doit] unwind to caller

doit:
  %cl = catchpad within %cs []
  call void @g(i32 %phi)
  unreachable

unreachable:
  unreachable
}

; CHECK-LABEL: define void @test3(
; CHECK:  %X = zext i8 %A to i32
; CHECK:  %Y = zext i8 %B to i32
; CHECK:  %phi = phi i32 [ %X, %bb ], [ %Y, %cont ], [ %Y, %cont2 ]

declare void @foo()
declare token @llvm.experimental.gc.statepoint.p0f_isVoidf(i64, i32, void ()*, i32, i32, ...)

define void @test4(i8 addrspace(1)* %obj) gc "statepoint-example" {
bb:
  unreachable

unreachable:
  call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* elementtype(void ()) @foo, i32 0, i32 0, i32 0, i32 0) ["deopt" (i32 0, i32 -1, i32 0, i32 0, i32 0)]
  ret void
}

; CHECK-LABEL: define void @test4(
; CHECK: unreachable:
; CHECK:   call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* elementtype(void ()) @foo, i32 0, i32 0, i32 0, i32 0) [ "deopt"(i32 0, i32 -1, i32 0, i32 0, i32 0) ]
; CHECK:   ret void


declare void @g(i32)
