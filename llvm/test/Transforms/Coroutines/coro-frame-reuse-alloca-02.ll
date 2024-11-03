; Tests that variables of different type in a Corotuine whose lifetime range is not overlapping each other
; re-use the same slot in Coroutine frame.
; RUN: opt -opaque-pointers=0 < %s -passes='cgscc(coro-split<reuse-storage>),simplifycfg,early-cse' -S | FileCheck %s
%"struct.task::promise_type" = type { i8 }
%struct.awaitable = type { i8 }
%struct.big_structure = type { [500 x i8] }
%struct.big_structure.2 = type { [300 x i8] }
declare i8* @malloc(i64)
declare void @consume(%struct.big_structure*)
declare void @consume.2(%struct.big_structure.2*)
define void @a(i1 zeroext %cond) presplitcoroutine {
entry:
  %__promise = alloca %"struct.task::promise_type", align 1
  %a = alloca %struct.big_structure, align 1
  %ref.tmp7 = alloca %struct.awaitable, align 1
  %b = alloca %struct.big_structure.2, align 1
  %ref.tmp18 = alloca %struct.awaitable, align 1
  %0 = getelementptr inbounds %"struct.task::promise_type", %"struct.task::promise_type"* %__promise, i64 0, i32 0
  %1 = call token @llvm.coro.id(i32 16, i8* nonnull %0, i8* bitcast (void (i1)* @a to i8*), i8* null)
  br label %init.ready
init.ready:
  %2 = call noalias nonnull i8* @llvm.coro.begin(token %1, i8* null)
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %0)
  br i1 %cond, label %if.then, label %if.else
if.then:
  %3 = getelementptr inbounds %struct.big_structure, %struct.big_structure* %a, i64 0, i32 0, i64 0
  call void @llvm.lifetime.start.p0i8(i64 500, i8* nonnull %3)
  call void @consume(%struct.big_structure* nonnull %a)
  %save = call token @llvm.coro.save(i8* null)
  %suspend = call i8 @llvm.coro.suspend(token %save, i1 false)
  switch i8 %suspend, label %coro.ret [
    i8 0, label %await.ready
    i8 1, label %cleanup1
  ]
await.ready:
  call void @llvm.lifetime.end.p0i8(i64 500, i8* nonnull %3)
  br label %cleanup1
if.else:
  %4 = getelementptr inbounds %struct.big_structure.2, %struct.big_structure.2* %b, i64 0, i32 0, i64 0
  call void @llvm.lifetime.start.p0i8(i64 300, i8* nonnull %4)
  call void @consume.2(%struct.big_structure.2* nonnull %b)
  %save2 = call token @llvm.coro.save(i8* null)
  %suspend2 = call i8 @llvm.coro.suspend(token %save2, i1 false)
  switch i8 %suspend2, label %coro.ret [
    i8 0, label %await2.ready
    i8 1, label %cleanup2
  ]
await2.ready:
  call void @llvm.lifetime.end.p0i8(i64 300, i8* nonnull %4)
  br label %cleanup2
cleanup1:
  call void @llvm.lifetime.end.p0i8(i64 500, i8* nonnull %3)
  br label %cleanup
cleanup2:
  call void @llvm.lifetime.end.p0i8(i64 300, i8* nonnull %4)
  br label %cleanup
cleanup:
  call i8* @llvm.coro.free(token %1, i8* %2)
  br label %coro.ret
coro.ret:
  call i1 @llvm.coro.end(i8* null, i1 false)
  ret void
}
; CHECK:       %a.Frame = type { void (%a.Frame*)*, void (%a.Frame*)*, %"struct.task::promise_type", %struct.big_structure, i1 }
; CHECK-LABEL: @a.resume(
; CHECK:         %[[A:.*]] = getelementptr inbounds %a.Frame, %a.Frame* %FramePtr, i32 0, i32 3
; CHECK:         %{{.*}} = bitcast %struct.big_structure* %[[A]] to %struct.big_structure.2*

declare token @llvm.coro.id(i32, i8* readnone, i8* nocapture readonly, i8*)
declare i1 @llvm.coro.alloc(token) #3
declare i64 @llvm.coro.size.i64() #5
declare i8* @llvm.coro.begin(token, i8* writeonly) #3
declare token @llvm.coro.save(i8*) #3
declare i8* @llvm.coro.frame() #5
declare i8 @llvm.coro.suspend(token, i1) #3
declare i8* @llvm.coro.free(token, i8* nocapture readonly) #2
declare i1 @llvm.coro.end(i8*, i1) #3
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #4
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #4
