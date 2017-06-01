; RUN: opt -sink -S < %s | FileCheck %s

target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc"

define void @h() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  %call = call i32 @g(i32 1) readnone
  invoke void @_CxxThrowException(i8* null, i8* null) noreturn
          to label %unreachable unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %cs = catchswitch within none [label %catch] unwind to caller

catch:                                            ; preds = %catch.dispatch
  %cp = catchpad within %cs [i8* null, i32 64, i8* null]
  catchret from %cp to label %try.cont

try.cont:                                         ; preds = %catch
  call void @k(i32 %call)
  ret void

unreachable:                                      ; preds = %entry
  unreachable
}

declare x86_stdcallcc void @_CxxThrowException(i8*, i8*)

declare i32 @__CxxFrameHandler3(...)

declare i32 @g(i32) readnone

declare void @k(i32)

; CHECK-LABEL: define void @h(
; CHECK: call i32 @g(i32 1)
; CHECK-NEXT: invoke void @_CxxThrowException(
