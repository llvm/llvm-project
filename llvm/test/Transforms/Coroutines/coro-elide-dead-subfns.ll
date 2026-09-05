; Tests that when a coroutine frame is elided onto the stack, CoroElide
; replaces stores of the out-of-line subfunction pointers into the frame header
; with a non-null sentinel constant so that GlobalDCE can delete the unused
; out-of-line .resume and .cleanup subfunctions after inlining.
; RUN: opt < %s -passes='default<O2>' -S | FileCheck %s

; Verify that the out-of-line subfunctions are completely eliminated from the module.
; CHECK-NOT: define internal fastcc void @f.resume
; CHECK-NOT: define internal fastcc void @f.cleanup
; CHECK-NOT: define internal fastcc void @f.destroy
; CHECK-LABEL: define noundef i32 @caller()

define internal ptr @f() presplitcoroutine {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr @f, ptr null)
  %need.alloc = call i1 @llvm.coro.alloc(token %id)
  br i1 %need.alloc, label %dyn.alloc, label %begin

dyn.alloc:
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  br label %begin

begin:
  %phi = phi ptr [ null, %entry ], [ %alloc, %dyn.alloc ]
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %phi)
  %s0 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %s0, label %suspend [
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

define i32 @caller() {
entry:
  %hdl = call ptr @f()
  %subfn = call ptr @llvm.coro.subfn.addr(ptr %hdl, i8 0)
  call fastcc void %subfn(ptr %hdl)
  %destroy = call ptr @llvm.coro.subfn.addr(ptr %hdl, i8 1)
  call fastcc void %destroy(ptr %hdl)
  ret i32 0
}

declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare i1 @llvm.coro.alloc(token)
declare i32 @llvm.coro.size.i32()
declare ptr @llvm.coro.begin(token, ptr)
declare i8 @llvm.coro.suspend(token, i1)
declare ptr @llvm.coro.free(token, ptr)
declare void @llvm.coro.end(ptr, i1, token)
declare ptr @llvm.coro.subfn.addr(ptr, i8)
declare noalias ptr @malloc(i32)
declare void @free(ptr)
