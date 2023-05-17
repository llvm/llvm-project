; Check that we can handle edge splits leading into a landingpad
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: define internal fastcc void @h.resume(
define void @h(i1 %cond, i32 %x, i32 %y) presplitcoroutine personality i32 0 {
entry:
  %id = call token @llvm.coro.id(i32 16, ptr null, ptr null, ptr null)
  %size = tail call i64 @llvm.coro.size.i64()
  %alloc = call ptr @malloc(i64 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  %sp = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp, label %coro.ret [
    i8 0, label %resume
    i8 1, label %cleanup
  ]

resume:
  br i1 %cond, label %invoke1, label %invoke2

invoke1:
  invoke void @may_throw1()
          to label %coro.ret unwind label %pad.with.phi
invoke2:
  invoke void @may_throw2()
          to label %coro.ret unwind label %pad.with.phi

; Verify that we created cleanuppads on every edge and inserted a reload of the spilled value

; CHECK: pad.with.phi.from.invoke2:
; CHECK:   %0 = cleanuppad within none []
; CHECK:   %y.reload.addr = getelementptr inbounds %h.Frame, ptr %hdl, i32 0, i32 3
; CHECK:   %y.reload = load i32, ptr %y.reload.addr
; CHECK:   cleanupret from %0 unwind label %pad.with.phi

; CHECK: pad.with.phi.from.invoke1:
; CHECK:   %1 = cleanuppad within none []
; CHECK:   %x.reload.addr = getelementptr inbounds %h.Frame, ptr %hdl, i32 0, i32 2
; CHECK:   %x.reload = load i32, ptr %x.reload.addr
; CHECK:   cleanupret from %1 unwind label %pad.with.phi

; CHECK: pad.with.phi:
; CHECK:   %val = phi i32 [ %x.reload, %pad.with.phi.from.invoke1 ], [ %y.reload, %pad.with.phi.from.invoke2 ]
; CHECK:   %switch = catchswitch within none [label %catch] unwind to caller
pad.with.phi:
  %val = phi i32 [ %x, %invoke1 ], [ %y, %invoke2 ]
  %switch = catchswitch within none [label %catch] unwind to caller

catch:                                            ; preds = %catch.dispatch
  %pad = catchpad within %switch [ptr null, i32 64, ptr null]
  call void @use_val(i32 %val)
  catchret from %pad to label %coro.ret

cleanup:                                        ; preds = %invoke.cont15, %if.else, %if.then, %ehcleanup21, %init.suspend
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %coro.ret

coro.ret:
  call i1 @llvm.coro.end(ptr null, i1 false)
  ret void
}

; Function Attrs: argmemonly nounwind readonly
declare token @llvm.coro.id(i32, ptr readnone, ptr nocapture readonly, ptr)
declare noalias ptr @malloc(i64)
declare i64 @llvm.coro.size.i64()
declare ptr @llvm.coro.begin(token, ptr writeonly)

; Function Attrs: nounwind
declare token @llvm.coro.save(ptr)
declare i8 @llvm.coro.suspend(token, i1)

; Function Attrs: argmemonly nounwind
declare void @may_throw1()
declare void @may_throw2()

declare ptr @__cxa_begin_catch(ptr)

declare void @use_val(i32)
declare void @__cxa_end_catch()

; Function Attrs: nounwind
declare i1 @llvm.coro.end(ptr, i1)
declare void @free(ptr)
declare ptr @llvm.coro.free(token, ptr nocapture readonly)
