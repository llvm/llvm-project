; Tests that coro-split passes initialized values to coroutine frame allocator.
; RUN: opt < %s -passes='cgscc(coro-split),simplify-cfg,early-cse' -S | FileCheck %s

define i8* @f(i32 %argument) "coroutine.presplit"="1" {
entry:
  %argument.addr = alloca i32, align 4
  %incremented = add i32 %argument, 1
  store i32 %incremented, i32* %argument.addr, align 4
  %id = call token @llvm.coro.id(i32 0, i8* null, i8* null, i8* null)
  %need.alloc = call i1 @llvm.coro.alloc(token %id)
  br i1 %need.alloc, label %dyn.alloc, label %begin

dyn.alloc:
  %size = call i32 @llvm.coro.size.i32()
  %allocator_argument = load i32, i32* %argument.addr, align 4
  %alloc = call i8* @custom_alloctor(i32 %size, i32 %allocator_argument)
  br label %begin

begin:
  %phi = phi i8* [ null, %entry ], [ %alloc, %dyn.alloc ]
  %hdl = call i8* @llvm.coro.begin(token %id, i8* %phi)
  %print_argument = load i32, i32* %argument.addr, align 4
  call void @print(i32 %print_argument)
  %0 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %0, label %suspend [i8 0, label %resume
                                i8 1, label %cleanup]
resume:
  call void @print(i32 1)
  br label %cleanup

cleanup:
  %mem = call i8* @llvm.coro.free(token %id, i8* %hdl)
  call void @free(i8* %mem)
  br label %suspend
suspend:
  call i1 @llvm.coro.end(i8* %hdl, i1 0)
  ret i8* %hdl
}

; CHECK-LABEL: @f(
; CHECK: %argument.addr = alloca i32
; CHECK: %incremented = add i32 %argument, 1
; CHECK-NEXT: store i32 %incremented, i32* %argument.addr
; CHECK-LABEL: dyn.alloc:
; CHECK: %allocator_argument = load i32, i32* %argument.addr
; CHECK: %alloc = call i8* @custom_alloctor(i32 24, i32 %allocator_argument)
; CHECK-LABEL: begin:
; CHECK: %print_argument = load i32, i32* %argument.addr
; CHECK: call void @print(i32 %print_argument)

declare i8* @llvm.coro.free(token, i8*)
declare i32 @llvm.coro.size.i32()
declare i8  @llvm.coro.suspend(token, i1)
declare void @llvm.coro.resume(i8*)
declare void @llvm.coro.destroy(i8*)

declare token @llvm.coro.id(i32, i8*, i8*, i8*)
declare i1 @llvm.coro.alloc(token)
declare i8* @llvm.coro.begin(token, i8*)
declare i1 @llvm.coro.end(i8*, i1)

declare noalias i8* @custom_alloctor(i32, i32)
declare void @print(i32)
declare void @free(i8*)
