; Tests that coro-split passes initialized values to coroutine frame allocator.
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

define ptr @f(i32 %argument) presplitcoroutine {
entry:
  %argument.addr = alloca i32, align 4
  %incremented = add i32 %argument, 1
  store i32 %incremented, ptr %argument.addr, align 4
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %need.alloc = call i1 @llvm.coro.alloc(token %id)
  br i1 %need.alloc, label %dyn.alloc, label %begin

dyn.alloc:
  %size = call i32 @llvm.coro.size.i32()
  %allocator_argument = load i32, ptr %argument.addr, align 4
  %alloc = call ptr @custom_alloctor(i32 %size, i32 %allocator_argument)
  br label %begin

begin:
  %phi = phi ptr [ null, %entry ], [ %alloc, %dyn.alloc ]
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %phi)
  %print_argument = load i32, ptr %argument.addr, align 4
  call void @print(i32 %print_argument)
  %0 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %0, label %suspend [i8 0, label %resume
                                i8 1, label %cleanup]
resume:
  call void @print(i32 1)
  br label %cleanup

cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend
suspend:
  call void @llvm.coro.end(ptr %hdl, i1 0, token none)
  ret ptr %hdl
}

; CHECK-LABEL: @f(
; CHECK: %argument.addr = alloca i32
; CHECK: %incremented = add i32 %argument, 1
; CHECK-NEXT: store i32 %incremented, ptr %argument.addr
; CHECK-LABEL: dyn.alloc:
; CHECK: %allocator_argument = load i32, ptr %argument.addr
; CHECK: %alloc = call ptr @custom_alloctor(i32 24, i32 %allocator_argument)
; CHECK-LABEL: begin:
; CHECK: %print_argument = load i32, ptr %argument.addr
; CHECK: call void @print(i32 %print_argument)

declare ptr @llvm.coro.free(token, ptr)
declare i32 @llvm.coro.size.i32()
declare i8  @llvm.coro.suspend(token, i1)
declare void @llvm.coro.resume(ptr)
declare void @llvm.coro.destroy(ptr)

declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare i1 @llvm.coro.alloc(token)
declare ptr @llvm.coro.begin(token, ptr)
declare void @llvm.coro.end(ptr, i1, token)

declare noalias ptr @custom_alloctor(i32, i32)
declare void @print(i32)
declare void @free(ptr)
