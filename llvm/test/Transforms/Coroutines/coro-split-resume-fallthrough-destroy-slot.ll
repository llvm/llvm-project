; Tests that a switch coroutine resume clone which falls through to coro.end
; suppresses only the deallocation for a stack-elided frame. This matters for
; allocation elision: the frame slot contains the cleanup clone in that case.
;
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

define ptr @f() presplitcoroutine {
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
  call void @print(i32 0)
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
  call void @llvm.coro.end(ptr %hdl, i1 false, token none)
  ret ptr %hdl
}

define void @caller() presplitcoroutine {
entry:
  %ptr = call ptr @f() #0
  ret void
}

; CHECK-LABEL: define ptr @f(
; CHECK:         %[[NEED_ALLOC:.+]] = call i1 @llvm.coro.alloc(
; CHECK:         %[[DESTROY_OR_CLEANUP:.+]] = select i1 %[[NEED_ALLOC]], ptr @f.destroy, ptr @f.cleanup
; CHECK:         %[[DESTROY_ADDR:.+]] = getelementptr inbounds i8, ptr %hdl, i64 8
; CHECK-NEXT:    store ptr %[[DESTROY_OR_CLEANUP]], ptr %[[DESTROY_ADDR]]

; CHECK-LABEL: define internal void @f.resume(
; CHECK:         %[[DESTROY_ADDR:.+]] = getelementptr inbounds i8, ptr %hdl, i64 8
; CHECK-NEXT:    %[[DESTROY:.+]] = load ptr, ptr %[[DESTROY_ADDR]]
; CHECK-NEXT:    %[[IS_ELIDED:.+]] = icmp eq ptr %[[DESTROY]], @f.cleanup
; CHECK:         call void @print(i32 1)
; CHECK:         %[[CORO_FREE:.+]] = select i1 %[[IS_ELIDED]], ptr null, ptr %hdl
; CHECK-NEXT:    call void @free(ptr %[[CORO_FREE]])
; CHECK-NEXT:    ret void

; CHECK-LABEL: define internal void @f.destroy(
; CHECK:         call void @free(

; CHECK-LABEL: define internal void @f.cleanup(
; CHECK-NOT:     call void @free(
; CHECK:         ret void

; CHECK-LABEL: define internal ptr @f.noalloc(
; CHECK:         %[[NOALLOC_DESTROY_ADDR:.+]] = getelementptr inbounds i8, ptr %{{.+}}, i64 8
; CHECK-NEXT:    store ptr @f.cleanup, ptr %[[NOALLOC_DESTROY_ADDR]]

declare ptr @llvm.coro.free(token, ptr)
declare i32 @llvm.coro.size.i32()
declare i8 @llvm.coro.suspend(token, i1)
declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare i1 @llvm.coro.alloc(token)
declare ptr @llvm.coro.begin(token, ptr)
declare void @llvm.coro.end(ptr, i1, token)

declare noalias ptr @malloc(i32) allockind("alloc,uninitialized") "alloc-family"="malloc"
declare void @print(i32)
declare void @free(ptr) willreturn allockind("free") "alloc-family"="malloc"

attributes #0 = { coro_elide_safe }
