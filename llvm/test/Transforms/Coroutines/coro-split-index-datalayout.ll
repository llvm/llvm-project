; Tests that coro-split uses DataLayout::getSmallestLegalIntType for the
; coroutine suspend index type so that targets with -n32 do not emit sub-word
; stores/loads for the suspend index.
; RUN: opt < %s -passes='cgscc(coro-split)' -S | FileCheck %s

target datalayout = "e-p:32:32-n32"

define ptr @f() presplitcoroutine {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr @f, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  %s0 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %s0, label %suspend [
    i8 0, label %step1
    i8 1, label %cleanup
  ]

step1:
  %s1 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %s1, label %suspend [
    i8 0, label %resume
    i8 1, label %cleanup
  ]

resume:
  br label %cleanup

cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend

suspend:
  call void @llvm.coro.end(ptr %hdl, i1 0, token none)
  ret ptr %hdl
}

; Verify that the suspend index stored and loaded in the frame uses i32
; rather than i1/i2 when DataLayout specifies -n32.
; CHECK-LABEL: define ptr @f()
; CHECK: store i32 0, ptr %index.addr
; CHECK-LABEL: define internal void @f.resume(
; CHECK: %index = load i32, ptr %index.addr

declare noalias ptr @malloc(i32)
declare void @free(ptr)
