; Tests the PHI nodes in cleanuppads for catchswitch instructions are correctly
; split up.
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg<switch-range-to-icmp>,early-cse' -S | FileCheck %s

declare i32 @__CxxFrameHandler3(...)
define ptr @f2(i1 %val) presplitcoroutine personality ptr @__CxxFrameHandler3 {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %valueA = call i32 @f();
  %valueB = call i32 @f();
  %need.alloc = call i1 @llvm.coro.alloc(token %id)
  br i1 %need.alloc, label %dyn.alloc, label %dowork.0

dyn.alloc:
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  br label %dowork.0

dowork.0:
  %phi = phi ptr [ null, %entry ], [ %alloc, %dyn.alloc ]
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %phi)
  invoke void @print(i32 0)
    to label %checksuspend unwind label %catch.dispatch.1

checksuspend:
  %0 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %0, label %suspend [i8 0, label %dowork.1
                                i8 1, label %cleanup]

dowork.1:
  invoke void @print(i32 0)
    to label %checksuspend unwind label %catch.dispatch.1

cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend

suspend:
  call i1 @llvm.coro.end(ptr %hdl, i1 0, token none)
  ret ptr %hdl

catch.dispatch.1:
  %cs1 = catchswitch within none [label %handler1] unwind to caller
handler1:
  %h1 = catchpad within %cs1 [ptr null, i32 64, ptr null]
  invoke void @print(i32 2) [ "funclet"(token %h1) ]
          to label %catchret1 unwind label %catch.dispatch.2
catchret1:
  catchret from %h1 to label %cleanup

catch.dispatch.2:
  %cs2 = catchswitch within %h1 [label %handler2] unwind label %cleanup2
handler2:
  %h2 = catchpad within %cs2 [ptr null, i32 64, ptr null]
  invoke void @print(i32 3) [ "funclet"(token %h2) ]
          to label %cleanup unwind label %cleanup2
cleanup2:
  %cleanupval2 = phi i32 [%valueA, %catch.dispatch.2], [%valueB, %handler2]
  cleanuppad within %h1 []
  call void @print(i32 %cleanupval2)
  br label %cleanup

; Verifiers that a "dispatcher" cleanuppad is created.

; catchswitch and all associated catchpads are required to have the same unwind
; edge, but coro requires that PHI nodes are split up so that reload
; instructions can be generated, therefore we create a new "dispatcher"
; cleanuppad which forwards to individual blocks that contain the reload
; instructions per catchswitch/catchpad and then all branch back to the
; original cleanuppad block.

; CHECK: catch.dispatch.2:
; CHECK:   %cs2 = catchswitch within %h1 [label %handler2] unwind label %cleanup2.corodispatch

; CHECK: handler2:
; CHECK:   invoke void @print(i32 3)
; CHECK:           to label %cleanup unwind label %cleanup2.corodispatch

; CHECK: cleanup2.corodispatch:
; CHECK:   %1 = phi i8 [ 0, %handler2 ], [ 1, %catch.dispatch.2 ]
; CHECK:   %2 = cleanuppad within %h1 []
; CHECK:   %switch = icmp ult i8 %1, 1
; CHECK:   br i1 %switch, label %cleanup2.from.handler2, label %cleanup2.from.catch.dispatch.2

; CHECK: cleanup2.from.handler2:
; CHECK:   %valueB.reload = load i32, ptr %valueB.spill.addr, align 4
; CHECK:   br label %cleanup2

; CHECK: cleanup2.from.catch.dispatch.2:
; CHECK:   %valueA.reload = load i32, ptr %valueA.spill.addr, align 4
; CHECK:   br label %cleanup2

; CHECK: cleanup2:
; CHECK:   %cleanupval2 = phi i32 [ %valueA.reload, %cleanup2.from.catch.dispatch.2 ], [ %valueB.reload, %cleanup2.from.handler2 ]
; CHECK:   call void @print(i32 %cleanupval2)
; CHECK:   br label %cleanup
}

declare ptr @llvm.coro.free(token, ptr)
declare i32 @llvm.coro.size.i32()
declare i8  @llvm.coro.suspend(token, i1)
declare void @llvm.coro.resume(ptr)
declare void @llvm.coro.destroy(ptr)

declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare i1 @llvm.coro.alloc(token)
declare ptr @llvm.coro.begin(token, ptr)
declare i1 @llvm.coro.end(ptr, i1, token)

declare noalias ptr @malloc(i32)
declare void @print(i32)
declare void @free(ptr)

declare i32 @f()
