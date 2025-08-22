; Tests that variables in a Corotuine whose lifetime range is not overlapping each other
; re-use the same slot in Coroutine frame.
; RUN: opt < %s -passes='cgscc(coro-split<reuse-storage>),simplifycfg,early-cse' -S | FileCheck %s
%"struct.task::promise_type" = type { i8 }
%struct.awaitable = type { i8 }
%struct.big_structure = type { [500 x i8] }
declare ptr @malloc(i64)
declare void @consume(ptr)
define void @a(i1 zeroext %cond) presplitcoroutine {
entry:
  %__promise = alloca %"struct.task::promise_type", align 1
  %a = alloca %struct.big_structure, align 1
  %ref.tmp7 = alloca %struct.awaitable, align 1
  %b = alloca %struct.big_structure, align 1
  %ref.tmp18 = alloca %struct.awaitable, align 1
  %0 = call token @llvm.coro.id(i32 16, ptr nonnull %__promise, ptr @a, ptr null)
  br label %init.ready
init.ready:
  %1 = call noalias nonnull ptr @llvm.coro.begin(token %0, ptr null)
  call void @llvm.lifetime.start.p0(ptr nonnull %__promise)
  br i1 %cond, label %if.then, label %if.else
if.then:
  call void @llvm.lifetime.start.p0(ptr nonnull %a)
  call void @consume(ptr nonnull %a)
  %save = call token @llvm.coro.save(ptr null)
  %suspend = call i8 @llvm.coro.suspend(token %save, i1 false)
  switch i8 %suspend, label %coro.ret [
    i8 0, label %await.ready
    i8 1, label %cleanup1
  ]
await.ready:
  call void @llvm.lifetime.end.p0(ptr nonnull %a)
  br label %cleanup1
if.else:
  call void @llvm.lifetime.start.p0(ptr nonnull %b)
  call void @consume(ptr nonnull %b)
  %save2 = call token @llvm.coro.save(ptr null)
  %suspend2 = call i8 @llvm.coro.suspend(token %save2, i1 false)
  switch i8 %suspend2, label %coro.ret [
    i8 0, label %await2.ready
    i8 1, label %cleanup2
  ]
await2.ready:
  call void @llvm.lifetime.end.p0(ptr nonnull %b)
  br label %cleanup2
cleanup1:
  call void @llvm.lifetime.end.p0(ptr nonnull %a)
  br label %cleanup
cleanup2:
  call void @llvm.lifetime.end.p0(ptr nonnull %b)
  br label %cleanup
cleanup:
  call ptr @llvm.coro.free(token %0, ptr %1)
  br label %coro.ret
coro.ret:
  call i1 @llvm.coro.end(ptr null, i1 false, token none)
  ret void
}

; check that there is only one %struct.big_structure in the frame.
; CHECK: %a.Frame = type { ptr, ptr, %"struct.task::promise_type", %struct.big_structure, i1 }

declare token @llvm.coro.id(i32, ptr readnone, ptr nocapture readonly, ptr)
declare i1 @llvm.coro.alloc(token) #3
declare i64 @llvm.coro.size.i64() #5
declare ptr @llvm.coro.begin(token, ptr writeonly) #3
declare token @llvm.coro.save(ptr) #3
declare ptr @llvm.coro.frame() #5
declare i8 @llvm.coro.suspend(token, i1) #3
declare ptr @llvm.coro.free(token, ptr nocapture readonly) #2
declare i1 @llvm.coro.end(ptr, i1, token) #3
declare void @llvm.lifetime.start.p0(ptr nocapture) #4
declare void @llvm.lifetime.end.p0(ptr nocapture) #4
