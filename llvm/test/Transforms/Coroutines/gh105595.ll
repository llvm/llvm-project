; Test that store-load operation that crosses suspension point will not be eliminated by DSE
; Coro result conversion function that attempts to modify promise shall produce this pattern
; RUN: opt < %s -passes='coro-early,dse' -S | FileCheck %s

define void @fn() presplitcoroutine {
  %__promise = alloca ptr, align 8
  %id = call token @llvm.coro.id(i32 16, ptr %__promise, ptr @fn, ptr null)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr null)
; CHECK: %promise.addr = call ptr @llvm.coro.promise(ptr %hdl, i32 8, i1 false)
  %save = call token @llvm.coro.save(ptr null)
  %sp = call i8 @llvm.coro.suspend(token %save, i1 false)
  %flag = icmp ule i8 %sp, 1
  br i1 %flag, label %resume, label %suspend

resume:
; CHECK: call void @use_value(ptr %promise.addr)
  call void @use_value(ptr %__promise)
  br label %suspend

suspend:
; load value when resume
; CHECK: %null = load ptr, ptr %promise.addr, align 8
  %null = load ptr, ptr %__promise, align 8
  call void @use_value(ptr %null)
; store value when suspend
; CHECK: store ptr null, ptr %promise.addr, align 8
  store ptr null, ptr %__promise, align 8
  ret void
}

declare void @use_value(ptr)
