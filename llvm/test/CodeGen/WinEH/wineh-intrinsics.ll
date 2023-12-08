; RUN: opt -passes=lint -disable-output < %s

; This test is meant to prove that the verifier does not report errors for correct
; use of the llvm.eh.exceptionpointer intrinsic.

target triple = "x86_64-pc-windows-msvc"

declare ptr @llvm.eh.exceptionpointer.p0(token)
declare ptr addrspace(1) @llvm.eh.exceptionpointer.p1(token)

declare void @f(...)

define void @test1() personality ptr @__CxxFrameHandler3 {
entry:
  invoke void (...) @f(i32 1)
     to label %exit unwind label %catchpad
catchpad:
  %cs1 = catchswitch within none [label %do_catch] unwind to caller
do_catch:
  %catch = catchpad within %cs1 [i32 1]
  %exn = call ptr @llvm.eh.exceptionpointer.p0(token %catch)
  call void (...) @f(ptr %exn)
  catchret from %catch to label %exit
exit:
  ret void
}

define void @test2() personality ptr @ProcessManagedException {
entry:
  invoke void (...) @f(i32 1)
     to label %exit unwind label %catchpad
catchpad:
  %cs1 = catchswitch within none [label %do_catch] unwind to caller
do_catch:
  %catch = catchpad within %cs1 [i32 1]
  %exn = call ptr addrspace(1) @llvm.eh.exceptionpointer.p1(token %catch)
  call void (...) @f(ptr addrspace(1) %exn)
  catchret from %catch to label %exit
exit:
  ret void
}

declare i32 @__CxxFrameHandler3(...)
declare i32 @ProcessManagedException(...)
