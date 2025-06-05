; Test that we rise lifetime markers to the correct place if there are many suspend points.
; RUN: opt < %s -passes='cgscc(coro-split),early-cse' -S | FileCheck %s

; CHECK: %large.alloca = alloca [500 x i8], align 16
; CHECK-NOT: %large.alloca.reload.addr

define void @many_suspend() presplitcoroutine {
entry:
  %large.alloca = alloca [500 x i8], align 16
  %id = call token @llvm.coro.id(i32 16, ptr null, ptr null, ptr null)
  %size = call i64 @llvm.coro.size.i64()
  %call = call noalias ptr @malloc(i64 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %call)
  call void @llvm.lifetime.start.p0(i64 500, ptr %large.alloca)
  %save1 = call token @llvm.coro.save(ptr null)
  %sp1 = call i8 @llvm.coro.suspend(token %save1, i1 false)
  switch i8 %sp1, label %coro.ret [
    i8 0, label %await.ready
    i8 1, label %cleanup
  ]

await.ready:
  %save2 = call token @llvm.coro.save(ptr null)
  %sp2 = call i8 @llvm.coro.suspend(token %save2, i1 false)
  switch i8 %sp2, label %coro.ret [
    i8 0, label %await2.ready
    i8 1, label %cleanup
  ]

await2.ready:
  %value = load i8, ptr %large.alloca, align 1
  call void @consume(i8 %value)
  %save3 = call token @llvm.coro.save(ptr null)
  %sp3 = call i8 @llvm.coro.suspend(token %save3, i1 false)
  switch i8 %sp3, label %coro.ret [
    i8 0, label %await3.ready
    i8 1, label %cleanup
  ]

await3.ready:
  %save4 = call token @llvm.coro.save(ptr null)
  %sp4 = call i8 @llvm.coro.suspend(token %save4, i1 false)
  switch i8 %sp4, label %coro.ret [
    i8 0, label %cleanup
    i8 1, label %cleanup
  ]

cleanup:
  call void @llvm.lifetime.end.p0(i64 500, ptr %large.alloca)
  %mem1 = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem1)
  br label %coro.ret

coro.ret:
  %InResumePart = call i1 @llvm.coro.end(ptr null, i1 false, token none)
  ret void
}

declare void @consume(i8)
declare ptr @malloc(i64)
declare void @free(ptr)
