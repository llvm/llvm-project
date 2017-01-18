; RUN: opt -S -inline < %s | FileCheck %s
; RUN: opt -S -passes='cgscc(inline)' < %s | FileCheck %s
target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc18.0.0"

declare void @g(i32)

define void @f() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @g(i32 0)
          to label %invoke.cont unwind label %cs.bb

invoke.cont:
  ret void

cs.bb:
  %cs = catchswitch within none [label %cp.bb] unwind label %cleanup.bb

cp.bb:
  %cpouter1 = catchpad within %cs [i8* null, i32 0, i8* null]
  call void @dtor() #1 [ "funclet"(token %cpouter1) ]
  catchret from %cpouter1 to label %invoke.cont

cleanup.bb:
  %cpouter2 = cleanuppad within none []
  call void @g(i32 1) [ "funclet"(token %cpouter2) ]
  cleanupret from %cpouter2 unwind to caller
}

declare i32 @__CxxFrameHandler3(...)

; Function Attrs: nounwind
define internal void @dtor() #1 personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @g(i32 2)
          to label %invoke.cont unwind label %ehcleanup1

invoke.cont:
  ret void

ehcleanup1:
  %cpinner1 = cleanuppad within none []
  invoke void @g(i32 3) [ "funclet" (token %cpinner1) ]
          to label %done unwind label %ehcleanup2
done:
  unreachable

ehcleanup2:
  %cpinner2 = cleanuppad within %cpinner1 []
  call void @g(i32 4) [ "funclet" (token %cpinner2) ]
  cleanupret from %cpinner2 unwind to caller
}

; CHECK-LABEL: define void @f(

; CHECK:      %[[cs:.*]] = catchswitch within none

; CHECK:      %[[cpouter1:.*]] = catchpad within %[[cs]]

; CHECK:      %[[cpinner1:.*]] = cleanuppad within %[[cpouter1]]

; CHECK:      %[[cpinner2:.*]] = cleanuppad within %[[cpinner1]]
; CHECK-NEXT: call void @g(i32 4) #0 [ "funclet"(token %[[cpinner2]]) ]
; CHECK-NEXT: unreachable

attributes #1 = { nounwind }
