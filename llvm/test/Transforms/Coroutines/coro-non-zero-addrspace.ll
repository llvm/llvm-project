; Verify Switch-ABI coroutine lowering, splitting, elision, and cleanup when the
; coroutine frame, promise, and allocas live in a non-zero address space (AS 5),
; with zero addrspacecasts emitted anywhere in the pipeline.
; RUN: opt < %s -passes='module(coro-early),cgscc(inline,coro-split,coro-elide,simplifycfg,early-cse),module(coro-cleanup)' -S | FileCheck %s

target datalayout = "e-p:64:64-A5"

declare void @print(i32)
declare noalias ptr addrspace(5) @alloc_as5(i32)
declare void @free_as5(ptr addrspace(5))

declare token @llvm.coro.id.p5(i32, ptr addrspace(5), ptr, ptr)
declare i1 @llvm.coro.alloc(token)
declare i32 @llvm.coro.size.i32()
declare ptr addrspace(5) @llvm.coro.begin.p5(token, ptr addrspace(5))
declare token @llvm.coro.save.p5(ptr addrspace(5))
declare i8 @llvm.coro.suspend(token, i1)
declare ptr addrspace(5) @llvm.coro.free.p5(token, ptr addrspace(5))
declare void @llvm.coro.dead.p5(ptr addrspace(5))
declare void @llvm.coro.end.p5(ptr addrspace(5), i1, token)
declare ptr addrspace(5) @llvm.coro.promise.p5(ptr addrspace(5), i32, i1)
declare void @llvm.coro.resume.p5(ptr addrspace(5))
declare void @llvm.coro.destroy.p5(ptr addrspace(5))
declare i1 @llvm.coro.done.p5(ptr addrspace(5))

define internal ptr addrspace(5) @coro_as5(i32 %x) presplitcoroutine {
entry:
  %promise = alloca i32, align 4, addrspace(5)
  %id = call token @llvm.coro.id.p5(i32 16, ptr addrspace(5) %promise, ptr null, ptr null)
  %need.alloc = call i1 @llvm.coro.alloc(token %id)
  br i1 %need.alloc, label %dyn.alloc, label %begin

dyn.alloc:
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr addrspace(5) @alloc_as5(i32 %size)
  br label %begin

begin:
  %phi = phi ptr addrspace(5) [ null, %entry ], [ %alloc, %dyn.alloc ]
  %hdl = call ptr addrspace(5) @llvm.coro.begin.p5(token %id, ptr addrspace(5) %phi)
  store i32 %x, ptr addrspace(5) %promise, align 4
  %s1 = call token @llvm.coro.save.p5(ptr addrspace(5) %hdl)
  %sp = call i8 @llvm.coro.suspend(token %s1, i1 false)
  switch i8 %sp, label %suspend [i8 0, label %resume
                                 i8 1, label %cleanup]

resume:
  %v = load i32, ptr addrspace(5) %promise, align 4
  call void @print(i32 %v)
  br label %cleanup

cleanup:
  %mem = call ptr addrspace(5) @llvm.coro.free.p5(token %id, ptr addrspace(5) %hdl)
  %need.free = icmp ne ptr addrspace(5) %mem, null
  br i1 %need.free, label %dyn.free, label %after.free

dyn.free:
  call void @free_as5(ptr addrspace(5) %mem)
  br label %after.free

after.free:
  call void @llvm.coro.dead.p5(ptr addrspace(5) %hdl)
  br label %suspend

suspend:
  call void @llvm.coro.end.p5(ptr addrspace(5) %hdl, i1 false, token none)
  ret ptr addrspace(5) %hdl
}

; CHECK-LABEL: define i32 @caller(i32 %x)
; CHECK-NOT: addrspacecast
; CHECK-NOT: @alloc_as5
; CHECK-NOT: @free_as5
; CHECK: %[[FRAME:[0-9a-zA-Z_.]+]] = alloca [{{[0-9]+}} x i8], align 8, addrspace(5)
; CHECK: store ptr @coro_as5.resume, ptr addrspace(5) %[[FRAME]], align 8
; CHECK: store ptr @coro_as5.cleanup, ptr addrspace(5) %{{.*}}, align 8
; CHECK: call void @coro_as5.resume(ptr addrspace(5) %[[FRAME]])
; CHECK: call void @coro_as5.cleanup(ptr addrspace(5) %[[FRAME]])
; CHECK: ret i32 %v
; CHECK-LABEL: define internal void @coro_as5.resume(ptr addrspace(5)
; CHECK-NOT: addrspacecast
; CHECK: call void @print(i32 %v)
define i32 @caller(i32 %x) {
entry:
  %hdl = call ptr addrspace(5) @coro_as5(i32 %x)
  %p = call ptr addrspace(5) @llvm.coro.promise.p5(ptr addrspace(5) %hdl, i32 4, i1 false)
  %v = load i32, ptr addrspace(5) %p, align 4
  call void @llvm.coro.resume.p5(ptr addrspace(5) %hdl)
  call void @llvm.coro.destroy.p5(ptr addrspace(5) %hdl)
  ret i32 %v
}
