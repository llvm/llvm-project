; Test we do not rise lifetime.end for allocas that may escape
; RUN: opt < %s -passes='cgscc(coro-split),early-cse' -S | FileCheck %s

; CHECK-NOT: %escape.gep = alloca [500 x i8], align 16
; CHECK: %escape.gep.reload.addr

; CHECK-NOT: %escape.store = alloca [500 x i8], align 16
; CHECK: %escape.store.reload.addr

; CHECK-NOT: %escape.call = alloca [500 x i8], align 16
; CHECK: %escape.call.reload.addr

define void @fn() presplitcoroutine {
entry:
  %id = call token @llvm.coro.id(i32 16, ptr null, ptr null, ptr null)
  %size = call i64 @llvm.coro.size.i64()
  %mem = call ptr @malloc(i64 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %mem)

  %escape.gep = alloca [500 x i8], align 16
  call void @llvm.lifetime.start.p0(i64 500, ptr %escape.gep)
  %gep.ptr = getelementptr inbounds nuw i8, ptr %escape.gep, i64 8

  %escape.store = alloca [500 x i8], align 16
  %store.ptr = alloca ptr, align 8
  call void @llvm.lifetime.start.p0(i64 500, ptr %escape.store)
  call void @llvm.lifetime.start.p0(i64 8, ptr %store.ptr)
  store ptr %escape.store, ptr %store.ptr, align 8

  %escape.call = alloca [500 x i8], align 16
  call void @llvm.lifetime.start.p0(i64 500, ptr %escape.call)
  call void @consume(ptr %escape.call)

  %save = call token @llvm.coro.save(ptr null)
  %suspend = call i8 @llvm.coro.suspend(token %save, i1 false)
  switch i8 %suspend, label %coro.ret [
    i8 0, label %await.ready
    i8 1, label %cleanup
  ]

await.ready:
  call void @consume(ptr %gep.ptr)
  call void @consume(ptr %store.ptr)
  br label %cleanup

cleanup:
  call void @llvm.lifetime.end.p0(i64 500, ptr %escape.gep)
  call void @llvm.lifetime.end.p0(i64 500, ptr %escape.store)
  call void @llvm.lifetime.end.p0(i64 500, ptr %escape.call)
  call void @llvm.lifetime.end.p0(i64 8, ptr %store.ptr)
  %mem1 = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem1)
  br label %coro.ret

coro.ret:
  %InResumePart = call i1 @llvm.coro.end(ptr null, i1 false, token none)
  ret void
}

declare void @consume(ptr)
declare ptr @malloc(i64)
declare void @free(ptr)
