; RUN: opt -lint -disable-output < %s

; This test is meant to prove that the verifier does not report errors for correct
; use of the llvm.eh.exceptionpointer intrinsic.

target triple = "x86_64-pc-windows-msvc"

declare i8* @llvm.eh.exceptionpointer.p0i8(token)
declare i8 addrspace(1)* @llvm.eh.exceptionpointer.p1i8(token)

declare void @f(...)

define void @test1() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void (...) @f(i32 1)
     to label %exit unwind label %catchpad
catchpad:
  %cs1 = catchswitch within none [label %do_catch] unwind to caller
do_catch:
  %catch = catchpad within %cs1 [i32 1]
  %exn = call i8* @llvm.eh.exceptionpointer.p0i8(token %catch)
  call void (...) @f(i8* %exn)
  catchret from %catch to label %exit
exit:
  ret void
}

define void @test2() personality i32 (...)* @ProcessManagedException {
entry:
  invoke void (...) @f(i32 1)
     to label %exit unwind label %catchpad
catchpad:
  %cs1 = catchswitch within none [label %do_catch] unwind to caller
do_catch:
  %catch = catchpad within %cs1 [i32 1]
  %exn = call i8 addrspace(1)* @llvm.eh.exceptionpointer.p1i8(token %catch)
  call void (...) @f(i8 addrspace(1)* %exn)
  catchret from %catch to label %exit
exit:
  ret void
}

declare i32 @__CxxFrameHandler3(...)
declare i32 @ProcessManagedException(...)
