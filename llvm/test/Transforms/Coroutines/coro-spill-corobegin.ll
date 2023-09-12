; Check that we can spills coro.begin from an inlined inner coroutine.
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

%g.Frame = type { ptr, ptr, i32, i1, i32 }

@g.resumers = private constant [3 x ptr] [ptr @g.dummy, ptr @g.dummy, ptr @g.dummy]

declare void @g.dummy(ptr)

declare ptr @g()

define ptr @f() presplitcoroutine {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)

  %innerid = call token @llvm.coro.id(i32 0, ptr null, ptr @g, ptr @g.resumers)
  %innerhdl = call noalias nonnull ptr @llvm.coro.begin(token %innerid, ptr null)

  %tok = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %tok, label %suspend [i8 0, label %resume
                                i8 1, label %cleanup]
resume:
  %gvar.addr = getelementptr inbounds %g.Frame, ptr %innerhdl, i32 0, i32 4
  %gvar = load i32, ptr %gvar.addr
  call void @print.i32(i32 %gvar)
  br label %cleanup

cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend
suspend:
  call i1 @llvm.coro.end(ptr %hdl, i1 0)
  ret ptr %hdl
}

; See if the ptr for coro.begin was added to f.Frame
; CHECK-LABEL: %f.Frame = type { ptr, ptr, ptr, i1 }

; See if the g's coro.begin was spilled into the frame
; CHECK-LABEL: @f(
; CHECK: %innerid = call token @llvm.coro.id(i32 0, ptr null, ptr @g, ptr @g.resumers)
; CHECK: %innerhdl = call noalias nonnull ptr @llvm.coro.begin(token %innerid, ptr null)
; CHECK: %[[spilladdr:.+]] = getelementptr inbounds %f.Frame, ptr %hdl, i32 0, i32 2
; CHECK: store ptr %innerhdl, ptr %[[spilladdr]]

; See if the coro.begin was loaded from the frame
; CHECK-LABEL: @f.resume(
; CHECK: %[[innerhdlAddr:.+]] = getelementptr inbounds %f.Frame, ptr %{{.+}}, i32 0, i32 2
; CHECK: %[[innerhdl:.+]] = load ptr, ptr %[[innerhdlAddr]]
; CHECK: %[[gvarAddr:.+]] = getelementptr inbounds %g.Frame, ptr %[[innerhdl]], i32 0, i32 4
; CHECK: %[[gvar:.+]] = load i32, ptr %[[gvarAddr]]
; CHECK: call void @print.i32(i32 %[[gvar]])

declare ptr @llvm.coro.free(token, ptr)
declare i32 @llvm.coro.size.i32()
declare i8  @llvm.coro.suspend(token, i1)

declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare i1 @llvm.coro.alloc(token)
declare ptr @llvm.coro.begin(token, ptr)
declare i1 @llvm.coro.end(ptr, i1)

declare noalias ptr @malloc(i32)
declare void @print.i32(i32)
declare void @free(ptr)
