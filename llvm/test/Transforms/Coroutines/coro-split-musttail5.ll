; Tests that sinked lifetime markers wouldn't provent optimization
; to convert a resuming call to a musttail call.
; RUN: opt < %s -passes='cgscc(coro-split<optimizing>),simplifycfg,early-cse' -S | FileCheck %s

declare void @fakeresume1(i64* align 8)

define void @g() #0 {
entry:
  %id = call token @llvm.coro.id(i32 0, i8* null, i8* null, i8* null)
  %alloc = call i8* @malloc(i64 16) #3
  %alloc.var = alloca i8
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %alloc.var)
  %vFrame = call noalias nonnull i8* @llvm.coro.begin(token %id, i8* %alloc)

  %save = call token @llvm.coro.save(i8* null)
  %suspend = call i8 @llvm.coro.suspend(token %save, i1 false)

  switch i8 %suspend, label %exit [
    i8 0, label %await.suspend
    i8 1, label %exit
  ]
await.suspend:
  %save2 = call token @llvm.coro.save(i8* null)
  call fastcc void @fakeresume1(i64* align 8 null)
  %suspend2 = call i8 @llvm.coro.suspend(token %save2, i1 false)
  switch i8 %suspend2, label %exit [
    i8 0, label %await.ready
    i8 1, label %exit
  ]
await.ready:
  call void @consume(i8* %alloc.var)
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %alloc.var)
  br label %exit
exit:
  call i1 @llvm.coro.end(i8* null, i1 false)
  ret void
}

; Verify that in the resume part resume call is marked with musttail.
; CHECK-LABEL: @g.resume(
; CHECK:          musttail call fastcc void @fakeresume1(i64* align 8 null)
; CHECK-NEXT:     ret void

declare token @llvm.coro.id(i32, i8* readnone, i8* nocapture readonly, i8*) #1
declare i1 @llvm.coro.alloc(token) #2
declare i64 @llvm.coro.size.i64() #3
declare i8* @llvm.coro.begin(token, i8* writeonly) #2
declare token @llvm.coro.save(i8*) #2
declare i8* @llvm.coro.frame() #3
declare i8 @llvm.coro.suspend(token, i1) #2
declare i8* @llvm.coro.free(token, i8* nocapture readonly) #1
declare i1 @llvm.coro.end(i8*, i1) #2
declare i8* @llvm.coro.subfn.addr(i8* nocapture readonly, i8) #1
declare i8* @malloc(i64)
declare void @consume(i8*)
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture)
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture)

attributes #0 = { presplitcoroutine }
attributes #1 = { argmemonly nounwind readonly }
attributes #2 = { nounwind }
attributes #3 = { nounwind readnone }
