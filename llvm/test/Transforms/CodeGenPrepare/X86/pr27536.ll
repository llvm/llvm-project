; RUN: opt -S -codegenprepare < %s | FileCheck %s
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

@rtti = external global i8

define void @test1() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  %e = alloca i8
  %tmpcast = bitcast i8* %e to i16*
  invoke void @_CxxThrowException(i8* null, i8* null)
          to label %catchret.dest unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %0 = catchswitch within none [label %catch] unwind to caller

catch:                                            ; preds = %catch.dispatch
  %1 = catchpad within %0 [i8* @rtti, i32 0, i16* %tmpcast]
  catchret from %1 to label %catchret.dest

catchret.dest:                                    ; preds = %catch
  ret void
}
; CHECK-LABEL: define void @test1(
; CHECK: %[[alloca:.*]] = alloca i8
; CHECK-NEXT: %[[bc:.*]] = bitcast i8* %[[alloca]] to i16*

; CHECK: catchpad within {{.*}} [i8* @rtti, i32 0, i16* %[[bc]]]

declare void @_CxxThrowException(i8*, i8*)

declare i32 @__CxxFrameHandler3(...)
