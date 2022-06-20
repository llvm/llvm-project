; Tests that coro-split will convert a call before coro.suspend to a musttail call
; while the user of the coro.suspend is a icmpinst.
; RUN: opt < %s -passes='cgscc(coro-split<optimizing>),simplifycfg,early-cse' -S | FileCheck %s

define void @fakeresume1(i8*)  {
entry:
  ret void;
}

define void @f() #0 {
entry:
  %id = call token @llvm.coro.id(i32 0, i8* null, i8* null, i8* null)
  %alloc = call i8* @malloc(i64 16) #3
  %vFrame = call noalias nonnull i8* @llvm.coro.begin(token %id, i8* %alloc)

  %save = call token @llvm.coro.save(i8* null)

  %init_suspend = call i8 @llvm.coro.suspend(token %save, i1 false)
  switch i8 %init_suspend, label %coro.end [
    i8 0, label %await.ready
    i8 1, label %coro.end
  ]
await.ready:
  %save2 = call token @llvm.coro.save(i8* null)

  call fastcc void @fakeresume1(i8* align 8 null)
  %suspend = call i8 @llvm.coro.suspend(token %save2, i1 true)
  %switch = icmp ult i8 %suspend, 2
  br i1 %switch, label %cleanup, label %coro.end

cleanup:
  %free.handle = call i8* @llvm.coro.free(token %id, i8* %vFrame)
  %.not = icmp eq i8* %free.handle, null
  br i1 %.not, label %coro.end, label %coro.free

coro.free:
  call void @delete(i8* nonnull %free.handle) #2
  br label %coro.end

coro.end:
  call i1 @llvm.coro.end(i8* null, i1 false)
  ret void
}

; CHECK-LABEL: @f.resume(
; CHECK:          musttail call fastcc void @fakeresume1(
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
declare void @delete(i8* nonnull) #2

attributes #0 = { presplitcoroutine }
attributes #1 = { argmemonly nounwind readonly }
attributes #2 = { nounwind }
attributes #3 = { nounwind readnone }
