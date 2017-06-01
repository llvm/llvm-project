; RUN: opt < %s -simplifycfg -S | FileCheck %s

declare void @f()
declare void @llvm.foo(i32) nounwind
declare void @ProcessCLRException()

define void @test1() personality void ()* @ProcessCLRException {
entry:
  invoke void @f()
    to label %exit unwind label %exn.dispatch
exn.dispatch:
  %cs = catchswitch within none [label %pad1, label %pad2] unwind to caller
pad1:
  %cp1 = catchpad within %cs [i32 1]
  call void @llvm.foo(i32 1)
  catchret from %cp1 to label %exit
pad2:
  %cp2 = catchpad within %cs [i32 2]
  unreachable
exit:
  ret void
}
; Remove unreachble catch2, leave catch1 as-is
; CHECK-LABEL: define void @test1()
; CHECK: %cs = catchswitch within none [label %pad1] unwind to caller
; CHECK-NOT: catchpad
; CHECK: %cp1 = catchpad within %cs [i32 1]
; CHECK-NOT: catchpad

; Remove both catchpads and the catchswitch from exn.dispatch
; CHECK-LABEL: define void @test2()
define void @test2() personality void ()* @ProcessCLRException {
entry:
  invoke void @f()
    to label %via.cleanup unwind label %exn.dispatch
  ; CHECK-NOT: invoke
  ; CHECK: call void @f()
via.cleanup:
  invoke void @f()
    to label %via.catchswitch unwind label %cleanup.inner
cleanup.inner:
  %cp.inner = cleanuppad within none []
  call void @llvm.foo(i32 0)
  cleanupret from %cp.inner unwind label %exn.dispatch
  ; CHECK: cleanupret from %cp.inner unwind to caller
via.catchswitch:
  invoke void @f()
    to label %exit unwind label %dispatch.inner
dispatch.inner:
  %cs.inner = catchswitch within none [label %pad.inner] unwind label %exn.dispatch
  ; CHECK: %cs.inner = catchswitch within none [label %pad.inner] unwind to caller
pad.inner:
  %catch.inner = catchpad within %cs.inner [i32 0]
  ; CHECK: %catch.inner = catchpad within %cs.inner
  call void @llvm.foo(i32 1)
  catchret from %catch.inner to label %exit
exn.dispatch:
  %cs = catchswitch within none [label %pad1, label %pad2] unwind to caller
  ; CHECK-NOT: catchswitch within
  ; CHECK-NOT: catchpad
pad1:
  catchpad within %cs [i32 1]
  unreachable
pad2:
  catchpad within %cs [i32 2]
  unreachable
exit:
  ret void
}

; Same as @test2, but exn.dispatch catchswitch has an unwind dest that
; preds need to be reidrected to
; CHECK-LABEL: define void @test3()
define void @test3() personality void ()* @ProcessCLRException {
entry:
  invoke void @f()
    to label %via.cleanup unwind label %exn.dispatch
  ; CHECK: invoke void @f()
  ; CHECK-NEXT: to label %via.cleanup unwind label %cleanup
via.cleanup:
  invoke void @f()
    to label %via.catchswitch unwind label %cleanup.inner
cleanup.inner:
  %cp.inner = cleanuppad within none []
  call void @llvm.foo(i32 0)
  cleanupret from %cp.inner unwind label %exn.dispatch
  ; CHECK: cleanupret from %cp.inner unwind label %cleanup
via.catchswitch:
  invoke void @f()
    to label %exit unwind label %dispatch.inner
dispatch.inner:
  %cs.inner = catchswitch within none [label %pad.inner] unwind label %exn.dispatch
  ; CHECK: %cs.inner = catchswitch within none [label %pad.inner] unwind label %cleanup
pad.inner:
  %catch.inner = catchpad within %cs.inner [i32 0]
  ; CHECK: %catch.inner = catchpad within %cs.inner
  call void @llvm.foo(i32 1)
  catchret from %catch.inner to label %exit
exn.dispatch:
  %cs = catchswitch within none [label %pad1, label %pad2] unwind label %cleanup
  ; CHECK-NOT: catchswitch within
  ; CHECK-NOT: catchpad
pad1:
  catchpad within %cs [i32 1]
  unreachable
pad2:
  catchpad within %cs [i32 2]
  unreachable
cleanup:
  %cp = cleanuppad within none []
  call void @llvm.foo(i32 0)
  cleanupret from %cp unwind to caller
exit:
  ret void
}
