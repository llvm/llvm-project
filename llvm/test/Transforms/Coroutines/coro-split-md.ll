; Tests that coro-split pass preserves metadata on suspend points.
; RUN: opt < %s -coro-split-preserves-suspend-md -passes='cgscc(coro-split)' -S | FileCheck %s

define ptr @f() presplitcoroutine {
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
  %0 = call i8 @llvm.coro.suspend(token none, i1 false), !llvm.coro.suspend_md !0
  switch i8 %0, label %suspend [i8 0, label %resume1
                                i8 1, label %cleanup]
resume1:
  call void @print(i32 1)
  %1 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %1, label %suspend [i8 0, label %resume2
                                i8 1, label %cleanup]
resume2:
  call void @print(i32 2)
  %2 = call i8 @llvm.coro.suspend(token none, i1 true), !llvm.coro.suspend_md !1
  switch i8 %2, label %suspend [i8 0, label %trap
                                i8 1, label %cleanup]
trap:
  call void @llvm.trap()
  unreachable
cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend
suspend:
  call i1 @llvm.coro.end(ptr %hdl, i1 0, token none)
  ret ptr %hdl
}

; CHECK: define ptr @f() !llvm.coro.suspend_md_table ![[MD_TABLE:[0-9]+]]

; CHECK-DAG: ![[MD_TABLE]] = !{![[S0_ROW:[0-9]+]], ![[S1_ROW:[0-9]+]]}
; CHECK-DAG: ![[S0_ROW]] = !{i2 0, ![[S0_MD:[0-9]+]]}
; CHECK-DAG: ![[S0_MD]] = !{!"waiting"}
; CHECK-DAG: ![[S1_ROW]] = !{i2 -2, ![[S1_MD:[0-9]+]]}
; CHECK-DAG: ![[S1_MD]] = !{!"done"}

declare ptr @llvm.coro.free(token, ptr)
declare i32 @llvm.coro.size.i32()
declare i8  @llvm.coro.suspend(token, i1)
declare void @llvm.coro.resume(ptr)
declare void @llvm.coro.destroy(ptr)

declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare i1 @llvm.coro.alloc(token)
declare ptr @llvm.coro.begin(token, ptr)
declare i1 @llvm.coro.end(ptr, i1, token)

declare noalias ptr @malloc(i32) allockind("alloc,uninitialized") "alloc-family"="malloc"
declare void @print(i32)
declare void @free(ptr) willreturn allockind("free") "alloc-family"="malloc"

!0 = !{!"waiting"}
!1 = !{!"done"}
attributes #1 = { coro_elide_safe }
