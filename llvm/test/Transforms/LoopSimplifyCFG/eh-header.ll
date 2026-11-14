; RUN: opt -S -passes='loop(loop-simplifycfg)' -verify-each -verify-loop-info -verify-dom-info -verify-loop-lcssa %s | FileCheck %s
; RUN: opt -S -passes='loop-mssa(loop-simplifycfg)' -verify-each -verify-memoryssa -verify-loop-info -verify-dom-info -verify-loop-lcssa %s | FileCheck %s
; RUN: opt -passes='default<O2>' -verify-each -disable-output %s

declare i32 @__CxxFrameHandler3(...)
declare void @may_throw()

; LoopSimplify cannot split the catchswitch's predecessors to make a normal
; preheader. Constant terminator folding must leave this valid EH loop alone.
; CHECK-LABEL: define void @catchswitch_header(
; CHECK: dispatch:
; CHECK-NEXT: %cs = catchswitch within none [label %handler] unwind to caller
; CHECK: handler:
; CHECK-NEXT: %cp = catchpad within %cs [ptr null, i32 64, ptr null]
; CHECK-NEXT: br i1 true, label %leave, label %dead
; CHECK: leave:
; CHECK-NEXT: catchret from %cp to label %retry
; CHECK: retry:
; CHECK-NEXT: invoke void @may_throw()
; CHECK-NEXT: to label %{{[^ ]+}} unwind label %dispatch
define void @catchswitch_header() personality ptr @__CxxFrameHandler3 {
entry:
  invoke void @may_throw() to label %done unwind label %dispatch

dispatch:
  %cs = catchswitch within none [label %handler] unwind to caller

handler:
  %cp = catchpad within %cs [ptr null, i32 64, ptr null]
  br i1 true, label %leave, label %dead

leave:
  catchret from %cp to label %retry

retry:
  invoke void @may_throw() to label %done unwind label %dispatch

dead:
  unreachable

done:
  ret void
}

; A cleanup before a catch is also a valid, unsplittable loop header when the
; catch returns to code that retries the throwing operation.
; CHECK-LABEL: define void @cleanuppad_header(
; CHECK: cleanup:
; CHECK-NEXT: %cp = cleanuppad within none []
; CHECK-NEXT: br i1 true, label %unwind, label %dead
; CHECK: unwind:
; CHECK-NEXT: cleanupret from %cp unwind label %dispatch
; CHECK: handler:
; CHECK-NEXT: %catch = catchpad within %cs [ptr null, i32 64, ptr null]
; CHECK-NEXT: catchret from %catch to label %retry
; CHECK: retry:
; CHECK-NEXT: invoke void @may_throw()
; CHECK-NEXT: to label %{{[^ ]+}} unwind label %cleanup
define void @cleanuppad_header() personality ptr @__CxxFrameHandler3 {
entry:
  invoke void @may_throw() to label %done unwind label %cleanup

cleanup:
  %cp = cleanuppad within none []
  br i1 true, label %unwind, label %dead

unwind:
  cleanupret from %cp unwind label %dispatch

dispatch:
  %cs = catchswitch within none [label %handler] unwind to caller

handler:
  %catch = catchpad within %cs [ptr null, i32 64, ptr null]
  catchret from %catch to label %retry

retry:
  invoke void @may_throw() to label %done unwind label %cleanup

dead:
  unreachable

done:
  ret void
}
