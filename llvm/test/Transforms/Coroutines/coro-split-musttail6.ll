; Tests that sinked lifetime markers wouldn't provent optimization
; to convert a resuming call to a musttail call.
; The difference between this and coro-split-musttail5.ll is that there is
; an extra bitcast instruction in the path, which makes it harder to
; optimize.
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s
; RUN: opt < %s -passes='pgo-instr-gen,cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

declare void @fakeresume1(ptr align 8)

define void @g() #0 {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %alloc = call ptr @malloc(i64 16) #3
  %alloc.var = alloca i64
  call void @llvm.lifetime.start.p0(i64 1, ptr %alloc.var)
  %vFrame = call noalias nonnull ptr @llvm.coro.begin(token %id, ptr %alloc)

  %save = call token @llvm.coro.save(ptr null)
  %suspend = call i8 @llvm.coro.suspend(token %save, i1 false)

  switch i8 %suspend, label %exit [
    i8 0, label %await.suspend
    i8 1, label %exit
  ]
await.suspend:
  %save2 = call token @llvm.coro.save(ptr null)
  call fastcc void @fakeresume1(ptr align 8 null)
  %suspend2 = call i8 @llvm.coro.suspend(token %save2, i1 false)
  switch i8 %suspend2, label %exit [
    i8 0, label %await.ready
    i8 1, label %exit
  ]
await.ready:
  call void @consume(ptr %alloc.var)
  call void @llvm.lifetime.end.p0(i64 1, ptr %alloc.var)
  br label %exit
exit:
  call i1 @llvm.coro.end(ptr null, i1 false, token none)
  ret void
}

; Verify that in the resume part resume call is marked with musttail.
; CHECK-LABEL: @g.resume(
; CHECK:      musttail call fastcc void @fakeresume1(ptr align 8 null)
; CHECK-NEXT: ret void

; It has a cleanup bb.
define void @f() #0 {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %alloc = call ptr @malloc(i64 16) #3
  %alloc.var = alloca i64
  call void @llvm.lifetime.start.p0(i64 1, ptr %alloc.var)
  %vFrame = call noalias nonnull ptr @llvm.coro.begin(token %id, ptr %alloc)

  %save = call token @llvm.coro.save(ptr null)
  %suspend = call i8 @llvm.coro.suspend(token %save, i1 false)

  switch i8 %suspend, label %exit [
    i8 0, label %await.suspend
    i8 1, label %exit
  ]
await.suspend:
  %save2 = call token @llvm.coro.save(ptr null)
  call fastcc void @fakeresume1(ptr align 8 null)
  %suspend2 = call i8 @llvm.coro.suspend(token %save2, i1 false)
  switch i8 %suspend2, label %exit [
    i8 0, label %await.ready
    i8 1, label %cleanup
  ]
await.ready:
  call void @consume(ptr %alloc.var)
  call void @llvm.lifetime.end.p0(i64 1, ptr %alloc.var)
  br label %exit

cleanup:
  %free.handle = call ptr @llvm.coro.free(token %id, ptr %vFrame)
  %.not = icmp eq ptr %free.handle, null
  br i1 %.not, label %exit, label %coro.free

coro.free:
  call void @delete(ptr nonnull %free.handle) #2
  br label %exit

exit:
  call i1 @llvm.coro.end(ptr null, i1 false, token none)
  ret void
}

; Verify that in the resume part resume call is marked with musttail.
; CHECK-LABEL: @f.resume(
; CHECK:      musttail call fastcc void @fakeresume1(ptr align 8 null)
; CHECK-NEXT: ret void

declare token @llvm.coro.id(i32, ptr readnone, ptr nocapture readonly, ptr) #1
declare i1 @llvm.coro.alloc(token) #2
declare i64 @llvm.coro.size.i64() #3
declare ptr @llvm.coro.begin(token, ptr writeonly) #2
declare token @llvm.coro.save(ptr) #2
declare ptr @llvm.coro.frame() #3
declare i8 @llvm.coro.suspend(token, i1) #2
declare ptr @llvm.coro.free(token, ptr nocapture readonly) #1
declare i1 @llvm.coro.end(ptr, i1, token) #2
declare ptr @llvm.coro.subfn.addr(ptr nocapture readonly, i8) #1
declare ptr @malloc(i64)
declare void @delete(ptr nonnull) #2
declare void @consume(ptr)
declare void @llvm.lifetime.start.p0(i64, ptr nocapture)
declare void @llvm.lifetime.end.p0(i64, ptr nocapture)

attributes #0 = { presplitcoroutine }
attributes #1 = { argmemonly nounwind readonly }
attributes #2 = { nounwind }
attributes #3 = { nounwind readnone }
