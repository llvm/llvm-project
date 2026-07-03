; RUN: llc < %s | FileCheck %s

target triple = "i386-pc-windows-msvc"

; Check that codegen doesn't fail due to wineh inserting instructions between
; the musttail call and return instruction.


define void @test() personality ptr @__CxxFrameHandler3 {
; CHECK-LABEL: test:

entry:
  invoke void @foo() to label %try.cont unwind label %catch.dispatch

catch.dispatch:
  %0 = catchswitch within none [label %catch] unwind to caller

catch:
  %1 = catchpad within %0 [ptr null, i32 64, ptr null]
  catchret from %1 to label %try.cont

try.cont:
; CHECK: movl %{{[a-z0-9]+}}, %fs:0
; CHECK: jmp _bar

  musttail call void @bar()
  ret void
}

declare i32 @__CxxFrameHandler3(...)
declare void @foo()
declare void @bar()
