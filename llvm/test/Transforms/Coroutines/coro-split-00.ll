; Tests that coro-split pass splits the coroutine into f, f.resume and f.destroy
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

define ptr @f() presplitcoroutine !func_sanitize !0 {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
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
  call void @llvm.coro.end(ptr %hdl, i1 0, token none)  
  ret ptr %hdl
}

; Make a safe_elide call to f and CoroSplit should generate the .noalloc variant
define void @caller() presplitcoroutine {
entry:
  %ptr = call ptr @f() #1
  ret void
}

; CHECK-LABEL: @f() !func_sanitize !0 {
; CHECK: call ptr @malloc
; CHECK: @llvm.coro.begin(token %id, ptr %phi)
; CHECK: store ptr @f.resume, ptr %hdl
; CHECK: %[[SEL:.+]] = select i1 %need.alloc, ptr @f.destroy, ptr @f.cleanup
; CHECK: store ptr %[[SEL]], ptr %destroy.addr
; CHECK: call void @print(i32 0)
; CHECK-NOT: call void @print(i32 1)
; CHECK-NOT: call void @free(
; CHECK: ret ptr %hdl

; CHECK-LABEL: @f.resume({{.*}}) {
; CHECK-NOT: call ptr @malloc
; CHECK-NOT: call void @print(i32 0)
; CHECK: call void @print(i32 1)
; CHECK-NOT: call void @print(i32 0)
; CHECK: call void @free(
; CHECK: ret void

; CHECK-LABEL: @f.destroy({{.*}}) {
; CHECK-NOT: call ptr @malloc
; CHECK-NOT: call void @print(
; CHECK: call void @free(
; CHECK: ret void

; CHECK-LABEL: @f.cleanup({{.*}}) {
; CHECK-NOT: call ptr @malloc
; CHECK-NOT: call void @print(
; CHECK-NOT: call void @free(
; CHECK: ret void

; CHECK-LABEL: @f.noalloc(ptr noundef nonnull align 8 dereferenceable(24) %{{.*}})
; CHECK-NOT: call ptr @malloc
; CHECK: call void @print(i32 0)
; CHECK-NOT: call void @print(i32 1)
; CHECK-NOT: call void @free(
; CHECK: ret ptr %{{.*}}

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

!0 = !{i32 846595819, ptr null}
attributes #1 = { coro_elide_safe }
