; Tests that instrumentation doesn't interfere with lowering (coro-split).
; It should convert coro.resume followed by a suspend to a musttail call.

; RUN: opt < %s -passes='pgo-instr-gen,cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

define void @fakeresume1(ptr)  {
entry:
  ret void;
}

define void @f() #0 {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %alloc = call ptr @malloc(i64 16) #3
  %vFrame = call noalias nonnull ptr @llvm.coro.begin(token %id, ptr %alloc)

  %save = call token @llvm.coro.save(ptr null)

  %init_suspend = call i8 @llvm.coro.suspend(token %save, i1 false)
  switch i8 %init_suspend, label %coro.end [
    i8 0, label %await.ready
    i8 1, label %coro.end
  ]
await.ready:
  %save2 = call token @llvm.coro.save(ptr null)

  call fastcc void @fakeresume1(ptr align 8 null)
  %suspend = call i8 @llvm.coro.suspend(token %save2, i1 true)
  %switch = icmp ult i8 %suspend, 2
  br i1 %switch, label %cleanup, label %coro.end

cleanup:
  %free.handle = call ptr @llvm.coro.free(token %id, ptr %vFrame)
  %.not = icmp eq ptr %free.handle, null
  br i1 %.not, label %coro.end, label %coro.free

coro.free:
  call void @delete(ptr nonnull %free.handle) #2
  br label %coro.end

coro.end:
  call i1 @llvm.coro.end(ptr null, i1 false, token none)
  ret void
}

; CHECK-LABEL: @f.resume(
; CHECK:          musttail call fastcc void @fakeresume1(
; CHECK-NEXT:     ret void

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

attributes #0 = { presplitcoroutine }
attributes #1 = { argmemonly nounwind readonly }
attributes #2 = { nounwind }
attributes #3 = { nounwind readnone }
