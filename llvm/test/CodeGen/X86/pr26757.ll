; RUN: llc < %s | FileCheck %s
target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i386-pc-windows-msvc"

declare void @throw()

define void @test1() personality i32 (...)* @__CxxFrameHandler3 {
  %e = alloca i8, align 4
  invoke void @throw()
          to label %.noexc unwind label %catch.dispatch

.noexc:
  unreachable

catch.object.Exception:
  %cp = catchpad within %cs [i8* null, i32 0, i8* %e]
  catchret from %cp to label %catchhandler

catch.dispatch:
  %cs = catchswitch within none [label %catch.object.Exception] unwind to caller

catchhandler:
  call void @use(i8* %e)
  ret void
}

; CHECK-LABEL: $handlerMap$0$test1:
; CHECK:      .long 0
; CHECK-NEXT: .long 0
; CHECK-NEXT: .long -20

declare void @use(i8*)

declare i32 @__CxxFrameHandler3(...)
