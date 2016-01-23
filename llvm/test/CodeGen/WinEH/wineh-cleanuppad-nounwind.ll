; RUN: opt -S -winehprepare < %s | FileCheck %s
target triple = "x86_64-pc-windows-msvc"

; CHECK-LABEL: @test1(
define void @test1(i1 %b) personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @f()
          to label %try.cont unwind label %cleanup.bb

; CHECK: entry:

; CHECK: [[catchswitch_entry:.*]]:
; CHECK-NEXT: %[[cs0:.*]] = catchswitch within none [label %[[catchpad:.*]]] unwind to caller
; CHECK: [[catchpad]]:
; CHECK-NEXT: %[[cp0:.*]] = catchpad within %[[cs0]] [i8* null, i32 64, i8* null]
; CHECK-NEXT: unreachable

try.cont:
  invoke void @f()
          to label %exit unwind label %catchswitch.bb

cleanup.bb:
  %cleanup = cleanuppad within none []
  br i1 %b, label %left, label %right

left:
  call void @exit(i32 0)  [ "funclet"(token %cleanup) ]
  unreachable

right:
  call void @exit(i32 1)  [ "funclet"(token %cleanup) ]
  unreachable

catchswitch.bb:
  %cs = catchswitch within none [label %catchpad.bb] unwind to caller

; CHECK: catchpad.bb:
; CHECK-NEXT:  %catch = catchpad within %cs [i8* null, i32 64, i8* null]

; CHECK: [[catchswitch_catch:.*]]:
; CHECK-NEXT: %[[cs1:.*]] = catchswitch within %catch [label %[[catchpad_catch:.*]]] unwind to caller
; CHECK: [[catchpad_catch]]:
; CHECK-NEXT: %[[cp1:.*]] = catchpad within %[[cs1]] [i8* null, i32 64, i8* null]
; CHECK-NEXT: unreachable

; CHECK: nested.cleanup.bb:
; CHECK-NEXT:  %nested.cleanup = cleanuppad within %catch []
; CHECK-NEXT:  call void @exit(i32 2)  [ "funclet"(token %nested.cleanup) ]
; CHECK-NEXT:  cleanupret from %nested.cleanup unwind label %[[catchswitch_catch]]

catchpad.bb:
  %catch = catchpad within %cs [i8* null, i32 64, i8* null]
  invoke void @f() [ "funclet"(token %catch) ]
          to label %unreachable unwind label %nested.cleanup.bb

nested.cleanup.bb:
  %nested.cleanup = cleanuppad within %catch []
  call void @exit(i32 2)  [ "funclet"(token %nested.cleanup) ]
  unreachable

unreachable:
  unreachable

exit:
  unreachable
}

declare void @f()
declare void @exit(i32) nounwind noreturn

declare i32 @__CxxFrameHandler3(...)
