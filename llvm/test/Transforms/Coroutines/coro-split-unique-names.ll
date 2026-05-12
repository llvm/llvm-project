; Tests that coro-split pass appends the .__uniq.<hash> suffix correctly at the end of split functions
; RUN: opt < %s -passes='cgscc(coro-split)' -S | FileCheck %s

; External linkage coroutine
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
  %0 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %0, label %suspend [i8 0, label %resume 
                                i8 1, label %cleanup]
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

; Internal linkage coroutine that already contains unique suffix
define internal ptr @g.__uniq.123456789() presplitcoroutine {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr @g.__uniq.123456789, ptr null)
  %need.alloc = call i1 @llvm.coro.alloc(token %id)
  br i1 %need.alloc, label %dyn.alloc, label %begin

dyn.alloc:  
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  br label %begin

begin:
  %phi = phi ptr [ null, %entry ], [ %alloc, %dyn.alloc ]
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %phi)
  %0 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %0, label %suspend [i8 0, label %resume 
                                i8 1, label %cleanup]
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

define void @caller() presplitcoroutine {
entry:
  %ptr1 = call ptr @f()
  %ptr2 = call ptr @g.__uniq.123456789()
  ret void
}

; CHECK-LABEL: define ptr @f()
; CHECK: store ptr @f.resume.__uniq.123456789, ptr %hdl

; CHECK-LABEL: define internal ptr @g.__uniq.123456789()
; CHECK: store ptr @g.resume.__uniq.123456789, ptr %hdl

; CHECK-LABEL: define internal fastcc void @f.resume.__uniq.123456789(
; CHECK-LABEL: define internal fastcc void @f.destroy.__uniq.123456789(
; CHECK-LABEL: define internal fastcc void @f.cleanup.__uniq.123456789(

; CHECK-LABEL: define internal fastcc void @g.resume.__uniq.123456789(
; CHECK-LABEL: define internal fastcc void @g.destroy.__uniq.123456789(
; CHECK-LABEL: define internal fastcc void @g.cleanup.__uniq.123456789(

declare ptr @llvm.coro.free(token, ptr)
declare i32 @llvm.coro.size.i32()
declare i8  @llvm.coro.suspend(token, i1)
declare void @llvm.coro.resume(ptr)
declare void @llvm.coro.destroy(ptr)
declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare i1 @llvm.coro.alloc(token)
declare ptr @llvm.coro.begin(token, ptr)
declare void @llvm.coro.end(ptr, i1, token) 
declare noalias ptr @malloc(i32)
declare void @free(ptr)

!llvm.module.flags = !{!0}
!0 = !{i32 4, !"ModuleNameHash", !"123456789"}
