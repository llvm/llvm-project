; Test auto-upgrading legacy unmangled Switch-ABI coroutine intrinsics to
; address-space-overloaded `.p0` / `.p0.p0` forms, including preserving
; operand bundles on `@llvm.coro.end`.
; RUN: llvm-as < %s | llvm-dis | FileCheck %s

define void @test_upgrade(ptr %p, ptr %fn, ptr %awaiter) personality i32 0 {
entry:
  %promise = alloca i32, align 4
  %id = call token @llvm.coro.id(i32 16, ptr %promise, ptr @test_upgrade, ptr null)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %p)
  %hdl2 = call ptr @llvm.coro.begin.custom.abi(token %id, ptr %p, i32 0)
  %save = call token @llvm.coro.save(ptr %hdl)
  call void @llvm.coro.await.suspend.void(ptr %awaiter, ptr %hdl, ptr %fn)
  %b = call i1 @llvm.coro.await.suspend.bool(ptr %awaiter, ptr %hdl, ptr %fn)
  call void @llvm.coro.await.suspend.handle(ptr %awaiter, ptr %hdl, ptr %fn)
  %pr = call ptr @llvm.coro.promise(ptr %hdl, i32 8, i1 false)
  call void @llvm.coro.resume(ptr %hdl)
  call void @llvm.coro.destroy(ptr %hdl)
  %done = call i1 @llvm.coro.done(ptr %hdl)
  %sub = call ptr @llvm.coro.subfn.addr(ptr %hdl, i8 0)
  %frm = call ptr @llvm.coro.frame()
  %noop = call ptr @llvm.coro.noop()
  %free = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @llvm.coro.dead(ptr %hdl)
  call void @llvm.coro.end(ptr %hdl, i1 false, token none)
  invoke void @may_throw() to label %exit unwind label %lpad

lpad:
  %tok = cleanuppad within none []
  call void @llvm.coro.end(ptr null, i1 true, token none) [ "funclet"(token %tok) ]
  cleanupret from %tok unwind to caller

exit:
  ret void
}

; CHECK-LABEL: define void @test_upgrade(
; CHECK: call token @llvm.coro.id.p0(i32 16, ptr %promise, ptr @test_upgrade, ptr null)
; CHECK: call ptr @llvm.coro.begin.p0(token %id, ptr %p)
; CHECK: call ptr @llvm.coro.begin.custom.abi.p0(token %id, ptr %p, i32 0)
; CHECK: call token @llvm.coro.save.p0(ptr %hdl)
; CHECK: call void @llvm.coro.await.suspend.void.p0.p0(ptr %awaiter, ptr %hdl, ptr %fn)
; CHECK: call i1 @llvm.coro.await.suspend.bool.p0.p0(ptr %awaiter, ptr %hdl, ptr %fn)
; CHECK: call void @llvm.coro.await.suspend.handle.p0.p0(ptr %awaiter, ptr %hdl, ptr %fn)
; CHECK: call ptr @llvm.coro.promise.p0(ptr %hdl, i32 8, i1 false)
; CHECK: call void @llvm.coro.resume.p0(ptr %hdl)
; CHECK: call void @llvm.coro.destroy.p0(ptr %hdl)
; CHECK: call i1 @llvm.coro.done.p0(ptr %hdl)
; CHECK: call ptr @llvm.coro.subfn.addr.p0(ptr %hdl, i8 0)
; CHECK: call ptr @llvm.coro.frame.p0()
; CHECK: call ptr @llvm.coro.noop.p0()
; CHECK: call ptr @llvm.coro.free.p0(token %id, ptr %hdl)
; CHECK: call void @llvm.coro.dead.p0(ptr %hdl)
; CHECK: call void @llvm.coro.end.p0(ptr %hdl, i1 false, token none)
; CHECK: call void @llvm.coro.end.p0(ptr null, i1 true, token none) [ "funclet"(token %tok) ]

declare void @may_throw()
declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare ptr @llvm.coro.begin(token, ptr)
declare ptr @llvm.coro.begin.custom.abi(token, ptr, i32)
declare token @llvm.coro.save(ptr)
declare void @llvm.coro.await.suspend.void(ptr, ptr, ptr)
declare i1 @llvm.coro.await.suspend.bool(ptr, ptr, ptr)
declare void @llvm.coro.await.suspend.handle(ptr, ptr, ptr)
declare ptr @llvm.coro.promise(ptr, i32, i1)
declare void @llvm.coro.resume(ptr)
declare void @llvm.coro.destroy(ptr)
declare i1 @llvm.coro.done(ptr)
declare ptr @llvm.coro.subfn.addr(ptr, i8)
declare ptr @llvm.coro.frame()
declare ptr @llvm.coro.noop()
declare ptr @llvm.coro.free(token, ptr)
declare void @llvm.coro.dead(ptr)
declare void @llvm.coro.end(ptr, i1, token)
