; Test that the printf library call simplifier works correctly.
;
; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc18.0.0"

@.str = private unnamed_addr constant [2 x i8] c"\0A\00", align 1

define void @test1() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @_CxxThrowException(i8* null, i8* null)
          to label %unreachable unwind label %catch.dispatch

catch.dispatch:
  %cs = catchswitch within none [label %catch] unwind to caller

catch:
  %cp = catchpad within %cs [i8* null, i32 64, i8* null]
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str, i32 0, i32 0)) [ "funclet"(token %cp) ]
  catchret from %cp to label %try.cont

try.cont:
  ret void

unreachable:
  unreachable
}

; CHECK-DAG: define void @test1(
; CHECK: %[[CS:.*]] = catchswitch within none
; CHECK: %[[CP:.*]] = catchpad within %[[CS]] [i8* null, i32 64, i8* null]
; CHECK: call i32 @putchar(i32 10) [ "funclet"(token %[[CP]]) ]

declare void @_CxxThrowException(i8*, i8*)

declare i32 @__CxxFrameHandler3(...)

declare i32 @printf(i8*, ...)
