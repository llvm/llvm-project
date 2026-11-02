; RUN: opt < %s -passes='module(coro-early),cgscc(coro-split),simplifycfg' -S | FileCheck %s

define ptr @f() presplitcoroutine !prof !0 {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  br label %loop

loop:
  %n = phi i32 [ 0, %entry ], [ %n.next, %loop.latch ]
  %suspend = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %suspend, label %suspend.ret [
    i8 0, label %loop.latch
    i8 1, label %cleanup
  ]

loop.latch:
  %n.next = add i32 %n, 1
  %cmp = icmp ult i32 %n.next, 4
  br i1 %cmp, label %loop, label %final, !prof !1

final:
  %final.suspend = call i8 @llvm.coro.suspend(token none, i1 true)
  switch i8 %final.suspend, label %suspend.ret [
    i8 0, label %final.ready
    i8 1, label %cleanup
  ]

final.ready:
  br label %cleanup

cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend.ret

suspend.ret:
  call void @llvm.coro.end(ptr %hdl, i1 false, token none)
  ret ptr %hdl
}

; CHECK-LABEL: define internal void @f.destroy(
; CHECK: %[[RESUME:.+]] = load ptr, ptr %{{.+}}

; Null is the final suspend. That weight stays, and the other edge is the rest.
; CHECK: %[[NULL:.+]] = icmp eq ptr %[[RESUME]], null

; True is the final suspend. False is the loop.
; CHECK: br i1 %[[NULL]], label %[[FINAL:.+]], label %[[OTHER:.+]], !prof ![[PROF:[0-9]+]]

; fitWeights prints a weight at or above 2^31 as a signed 32-bit value.
; CHECK: ![[PROF]] = !{!"branch_weights", i32 504333280, i32 -2147483648}

declare ptr @llvm.coro.free(token, ptr)
declare i32 @llvm.coro.size.i32()
declare i8 @llvm.coro.suspend(token, i1)
declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare ptr @llvm.coro.begin(token, ptr)
declare void @llvm.coro.end(ptr, i1, token)
declare noalias ptr @malloc(i32)
declare void @free(ptr)

!0 = !{!"function_entry_count", i64 100}
; The loop continues 3 times and exits once.
!1 = !{!"branch_weights", i32 3, i32 1}
