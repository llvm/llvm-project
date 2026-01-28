; Tests that coro-split pass generates TBAA metadata on coroutine frame slot reloads.
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

; CHECK-LABEL: @f.resume(
; CHECK-SAME: %[[HDL:[A-Za-z0-9_]+]]
define ptr @f(ptr %p) presplitcoroutine {
entry:
  %x = load i32, ptr %p, !tbaa !3
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
  call void @print(i32 %x)
  ; CHECK: %[[X_RELOAD_ADDR:.+]] = getelementptr inbounds %f.Frame, ptr %[[HDL]], i32 0, i32 {{[0-9]+}}
  ; CHECK: %{{.+}} = load i32, ptr %[[X_RELOAD_ADDR]], align 4, !tbaa ![[COROUTINE_SLOT_TAG:[0-9]+]]
  br label %cleanup

cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend
suspend:
  call void @llvm.coro.end(ptr %hdl, i1 0, token none)
  ret ptr %hdl
}

; CHECK: ![[COROUTINE_SLOT_ROOT:.+]] = !{!"Simple C++ TBAA"}
; CHECK: ![[COROUTINE_SLOT_TAG]] = !{![[COROUTINE_SLOT_SCALAR:[0-9]+]], ![[COROUTINE_SLOT_SCALAR]], i64 0}
; CHECK: ![[COROUTINE_SLOT_SCALAR]] = !{!"f.Frame Slot", ![[COROUTINE_SLOT_ROOT]], i64 0}
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
