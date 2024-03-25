; Check that we can handle edge splits leading into a landingpad
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: define internal fastcc void @f.resume(
define void @f(i1 %cond) presplitcoroutine personality i32 0 {
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
          to label %unreach unwind label %pad.with.phi
invoke2:
  invoke void @may_throw2()
          to label %unreach unwind label %pad.with.phi

; Verify that we cloned landing pad on every edge and inserted a reload of the spilled value

; CHECK: pad.with.phi.from.invoke2:
; CHECK:   %0 = landingpad { ptr, i32 }
; CHECK:           catch ptr null
; CHECK:   br label %pad.with.phi

; CHECK: pad.with.phi.from.invoke1:
; CHECK:   %1 = landingpad { ptr, i32 }
; CHECK:           catch ptr null
; CHECK:   br label %pad.with.phi

; CHECK: pad.with.phi:
; CHECK:   %val = phi i32 [ 0, %pad.with.phi.from.invoke1 ], [ 1, %pad.with.phi.from.invoke2 ]
; CHECK:   %lp = phi { ptr, i32 } [ %0, %pad.with.phi.from.invoke2 ], [ %1, %pad.with.phi.from.invoke1 ]
; CHECK:   %exn = extractvalue { ptr, i32 } %lp, 0
; CHECK:   call ptr @__cxa_begin_catch(ptr %exn)
; CHECK:   call void @use_val(i32 %val)
; CHECK:   call void @__cxa_end_catch()
; CHECK:   call void @free(ptr %hdl)
; CHECK:   ret void

pad.with.phi:
  %val = phi i32 [ 0, %invoke1 ], [ 1, %invoke2 ]
  %lp = landingpad { ptr, i32 }
          catch ptr null
  %exn = extractvalue { ptr, i32 } %lp, 0
  call ptr @__cxa_begin_catch(ptr %exn)
  call void @use_val(i32 %val)
  call void @__cxa_end_catch()
  br label %cleanup

cleanup:                                        ; preds = %invoke.cont15, %if.else, %if.then, %ehcleanup21, %init.suspend
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %coro.ret

coro.ret:
  call i1 @llvm.coro.end(ptr null, i1 false, token none)
  ret void

unreach:
  unreachable
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
declare i1 @llvm.coro.end(ptr, i1, token)
declare void @free(ptr)
declare ptr @llvm.coro.free(token, ptr nocapture readonly)
