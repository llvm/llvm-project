; Tests that coro-split won't fall in infinite loop when simplify the terminators leading to ret.
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s
; RUN: opt < %s -passes='pgo-instr-gen,cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

declare void @fakeresume1(ptr)
declare void @may_throw(ptr)
declare void @print()

define void @f(i1 %cond) #0 personality i32 3 {
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
  call fastcc void @fakeresume1(ptr align 8 null)
  invoke void @may_throw(ptr null)
    to label %ready unwind label %lpad

ready:
  %save2 = call token @llvm.coro.save(ptr null)
  %suspend = call i8 @llvm.coro.suspend(token %save2, i1 true)
  %switch = icmp ult i8 %suspend, 2
  br i1 %switch, label %cleanup, label %coro.end

cleanup:
  %free.handle = call ptr @llvm.coro.free(token %id, ptr %vFrame)
  %.not = icmp eq ptr %free.handle, null
  br i1 %.not, label %coro.end, label %coro.free

lpad:
  %lpval = landingpad { ptr, i32 }
     cleanup

  %need.resume = call i1 @llvm.coro.end(ptr null, i1 true, token none)
  resume { ptr, i32 } %lpval

coro.free:
  call void @delete(ptr nonnull %free.handle) #2
  br label %coro.end

coro.end:
  call i1 @llvm.coro.end(ptr null, i1 false, token none)
  ret void
}

; CHECK-LABEL: @f.resume(
; CHECK-NOT:          musttail call fastcc void @fakeresume1(
; CHECK:     }


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
