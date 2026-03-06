; A negative test to check coro-split does not generate TBAA metadata on frame access
; in cases it should not:
; - Access to the promise slot
; - Access to alloca slots
; - Access to internally managed slots, such as the suspend point index
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

; CHECK-NOT: f.Frame Slot

@g_x1 = global i32 1

define ptr @f() presplitcoroutine {
entry:
  %promise = alloca i32
  %buffer = alloca [128 x i8]
  %x1.1 = load i32, ptr @g_x1, !tbaa !3
  %ref1 = getelementptr inbounds [128 x i8], ptr %buffer, i32 0, i32 %x1.1
  store i8 42, ptr %ref1
  %id = call token @llvm.coro.id(i32 0, ptr %promise, ptr null, ptr null)
  %need.alloc = call i1 @llvm.coro.alloc(token %id)
  br i1 %need.alloc, label %dyn.alloc, label %begin

dyn.alloc:
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  br label %begin

begin:
  %phi = phi ptr [ null, %entry ], [ %alloc, %dyn.alloc ]
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %phi)
  store i32 111, ptr %promise
  call void @print(i32 0)
  %0 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %0, label %suspend [i8 0, label %resume
                                i8 1, label %cleanup]
resume:
  %the_111_from_earlier = load i32, ptr %promise, !tbaa !3
  call void @print(i32 %the_111_from_earlier)
  %x1.2 = load i32, ptr @g_x1, !tbaa !3
  %ref2 = getelementptr inbounds [128 x i8], ptr %buffer, i32 0, i32 %x1.2
  %result = load i32, ptr %ref2, !tbaa !3
  store i32 %result, ptr %promise, !tbaa !3
  %1 = call i8 @llvm.coro.suspend(token none, i1 true)
  switch i8 %1, label %suspend [i8 1, label %cleanup]

cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend
suspend:
  call void @llvm.coro.end(ptr %hdl, i1 0, token none)
  ret ptr %hdl
}

!0 = !{!"Simple C++ TBAA"}
!1 = !{!"omnipotent char", !0, i64 0}
!2 = !{!"int", !1, i64 0}
!3 = !{!2, !2, i64 0}


declare ptr @llvm.coro.free(token, ptr)
declare i32 @llvm.coro.size.i32()
declare i8  @llvm.coro.suspend(token, i1)
declare void @llvm.coro.resume(ptr)
declare void @llvm.coro.destroy(ptr)

declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare i1 @llvm.coro.alloc(token)
declare ptr @llvm.coro.begin(token, ptr)
declare void @llvm.coro.end(ptr, i1, token)

declare noalias ptr @malloc(i32) allockind("alloc,uninitialized") "alloc-family"="malloc"
declare void @print(i32)
declare void @free(ptr) willreturn allockind("free") "alloc-family"="malloc"
