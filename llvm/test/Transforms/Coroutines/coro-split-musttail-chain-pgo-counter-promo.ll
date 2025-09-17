; REQUIRES: x86-registered-target
; RUN: opt -passes='pgo-instr-gen,instrprof,coro-split' -do-counter-promotion=true -S < %s | FileCheck %s

; CHECK-LABEL: define internal fastcc void @f.resume
; CHECK: musttail call fastcc void 
; CHECK-NEXT: ret void
; CHECK: musttail call fastcc void 
; CHECK-NEXT: ret void
; CHECK-LABEL: define internal fastcc void @f.destroy
target triple = "x86_64-grtev4-linux-gnu"

%CoroutinePromise = type { ptr, i64, [8 x i8], ptr} 
%Awaitable.1 = type { ptr }
%Awaitable.2 = type { ptr, ptr }

declare void @await_suspend(ptr noundef nonnull align 1 dereferenceable(1), ptr) local_unnamed_addr
declare ptr @await_transform_await_suspend(ptr noundef nonnull align 8 dereferenceable(16), ptr) local_unnamed_addr
declare void @destroy_frame_slowpath(ptr noundef nonnull align 16 dereferenceable(32)) local_unnamed_addr
declare ptr @other_coro();
declare void @heap_delete(ptr noundef, i64 noundef, i64 noundef) local_unnamed_addr
declare noundef nonnull ptr @heap_allocate(i64 noundef, i64 noundef) local_unnamed_addr

declare void @llvm.assume(i1 noundef)
declare i64 @llvm.coro.align.i64()
declare i1 @llvm.coro.alloc(token)
declare ptr @llvm.coro.begin(token, ptr writeonly)
declare void @llvm.coro.end(ptr, i1, token)
declare ptr @llvm.coro.free(token, ptr nocapture readonly)
declare token @llvm.coro.id(i32, ptr readnone, ptr nocapture readonly, ptr)
declare token @llvm.coro.save(ptr)
declare i64 @llvm.coro.size.i64()
declare ptr @llvm.coro.subfn.addr(ptr nocapture readonly, i8)
declare i8 @llvm.coro.suspend(token, i1)
declare void @llvm.instrprof.increment(ptr, i64, i32, i32)
declare void @llvm.instrprof.value.profile(ptr, i64, i64, i32, i32)
declare void @llvm.lifetime.start.p0(ptr nocapture)
declare void @llvm.lifetime.end.p0(ptr nocapture)

; Function Attrs: noinline nounwind presplitcoroutine uwtable
define ptr @f(i32 %0) presplitcoroutine align 32 {
  %2 = alloca i32, align 8
  %3 = alloca %CoroutinePromise, align 16
  %4 = alloca %Awaitable.1, align 8
  %5 = alloca %Awaitable.2, align 8
  %6 = call token @llvm.coro.id(i32 8, ptr nonnull %3, ptr nonnull @f, ptr null)
  %7 = call i1 @llvm.coro.alloc(token %6)
  br i1 %7, label %8, label %12

8:                                                ; preds = %1
  %9 = call i64 @llvm.coro.size.i64()
  %10 = call i64 @llvm.coro.align.i64()
  %11 = call noalias noundef nonnull ptr @heap_allocate(i64 noundef %9, i64 noundef %10) #27
  call void @llvm.assume(i1 true) [ "align"(ptr %11, i64 %10) ]
  br label %12

12:                                               ; preds = %8, %1
  %13 = phi ptr [ null, %1 ], [ %11, %8 ]
  %14 = call ptr @llvm.coro.begin(token %6, ptr %13) #28
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #9
  store ptr null, ptr %3, align 16
  %15 = getelementptr inbounds {ptr, i64}, ptr %3, i64 0, i32 1
  store i64 0, ptr %15, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #9
  store ptr %3, ptr %4, align 8
  %16 = call token @llvm.coro.save(ptr null)
  call void @await_suspend(ptr noundef nonnull align 1 dereferenceable(1) %4, ptr %14) #9
  %17 = call i8 @llvm.coro.suspend(token %16, i1 false)
  switch i8 %17, label %61 [
    i8 0, label %18
    i8 1, label %21
  ]

18:                                               ; preds = %12
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #9
  %19 = icmp slt i32 0, %0
  br i1 %19, label %20, label %36

20:                                               ; preds = %18
  br label %22

21:                                               ; preds = %12
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #9
  br label %54

22:                                               ; preds = %20, %31
  %23 = phi i32 [ 0, %20 ], [ %32, %31 ]
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #9
  %24 = call ptr @other_coro()
  store ptr %3, ptr %5, align 8
  %25 = getelementptr inbounds { ptr, ptr }, ptr %5, i64 0, i32 1
  store ptr %24, ptr %25, align 8
  %26 = call token @llvm.coro.save(ptr null)
  call void @llvm.coro.await.suspend.handle(ptr null, ptr null, ptr @await_transform_await_suspend)
  %30 = call i8 @llvm.coro.suspend(token %26, i1 false)
  switch i8 %30, label %60 [
    i8 0, label %31
    i8 1, label %34
  ]

31:                                               ; preds = %22
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #9
  %32 = add nuw nsw i32 %23, 1
  %33 = icmp slt i32 %32, %0
  br i1 %33, label %22, label %35, !llvm.loop !0

34:                                               ; preds = %22
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #9
  br label %54

35:                                               ; preds = %31
  br label %36

36:                                               ; preds = %35, %18
  %37 = call token @llvm.coro.save(ptr null)
  %38 = getelementptr inbounds i8, ptr %14, i64 16
  %39 = getelementptr inbounds i8, ptr %14, i64 32
  %40 = load i64, ptr %39, align 8
  %41 = load ptr, ptr %38, align 16
  %42 = icmp eq ptr %41, null
  br i1 %42, label %43, label %46

43:                                               ; preds = %36
  call void @llvm.coro.await.suspend.handle(ptr null, ptr null, ptr @await_transform_await_suspend)
  br label %47

46:                                               ; preds = %36
  call void @destroy_frame_slowpath(ptr noundef nonnull align 16 dereferenceable(32) %38) #9
  br label %47

47:                                               ; preds = %43, %46
  %48 = inttoptr i64 %40 to ptr
  %49 = call ptr @llvm.coro.subfn.addr(ptr %48, i8 0)
  %50 = ptrtoint ptr %49 to i64
  call fastcc void %49(ptr %48) #9
  %51 = call i8 @llvm.coro.suspend(token %37, i1 true) #28
  switch i8 %51, label %61 [
    i8 0, label %53
    i8 1, label %52
  ]

52:                                               ; preds = %47
  br label %54

53:                                               ; preds = %47
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #9
  unreachable

54:                                               ; preds = %52, %34, %21
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #9
  %55 = call ptr @llvm.coro.free(token %6, ptr %14)
  %56 = icmp eq ptr %55, null
  br i1 %56, label %61, label %57

57:                                               ; preds = %54
  %58 = call i64 @llvm.coro.size.i64()
  %59 = call i64 @llvm.coro.align.i64()
  call void @heap_delete(ptr noundef nonnull %55, i64 noundef %58, i64 noundef %59) #9
  br label %61

60:                                               ; preds = %22
  br label %61

61:                                               ; preds = %60, %57, %54, %47, %12
  %62 = getelementptr inbounds i8, ptr %3, i64 -16
  call void @llvm.coro.end(ptr null, i1 false, token none) #28
  ret ptr %62
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.mustprogress"}
