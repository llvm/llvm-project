; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

%"struct.std::coroutine_handle" = type { ptr }
%"struct.std::coroutine_handle.0" = type { %"struct.std::coroutine_handle" }
%"struct.lean_future<int>::Awaiter" = type { i32, %"struct.std::coroutine_handle.0" }

declare ptr @malloc(i64)

%i8.array = type { [100 x i8] }
declare void @consume.i8.array(ptr)

; The lifetime of testval starts and ends before coro.suspend. Even though consume.i8.array
; might capture it, we can safely say it won't live across suspension.
define void @foo() presplitcoroutine {
entry:
  %testval = alloca %i8.array
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %alloc = call ptr @malloc(i64 16) #3
  %vFrame = call noalias nonnull ptr @llvm.coro.begin(token %id, ptr %alloc)

  call void @llvm.lifetime.start.p0(i64 100, ptr %testval)
  call void @consume.i8.array(ptr %testval)
  call void @llvm.lifetime.end.p0(i64 100, ptr  %testval)

  %save = call token @llvm.coro.save(ptr null)
  %suspend = call i8 @llvm.coro.suspend(token %save, i1 false)
  switch i8 %suspend, label %exit [
    i8 0, label %await.ready
    i8 1, label %exit
  ]
await.ready:
  %StrayCoroSave = call token @llvm.coro.save(ptr null)
  br label %exit
exit:
  call i1 @llvm.coro.end(ptr null, i1 false)
  ret void
}

; The lifetime of testval starts after coro.suspend. So it will never live across suspension
; points.
define void @bar() presplitcoroutine {
entry:
  %testval = alloca %i8.array
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %alloc = call ptr @malloc(i64 16) #3
  %vFrame = call noalias nonnull ptr @llvm.coro.begin(token %id, ptr %alloc)
  %save = call token @llvm.coro.save(ptr null)
  %suspend = call i8 @llvm.coro.suspend(token %save, i1 false)
  switch i8 %suspend, label %exit [
    i8 0, label %await.ready
    i8 1, label %exit
  ]
await.ready:
  %StrayCoroSave = call token @llvm.coro.save(ptr null)

  call void @llvm.lifetime.start.p0(i64 100, ptr %testval)
  call void @consume.i8.array(ptr %testval)
  call void @llvm.lifetime.end.p0(i64 100, ptr  %testval)

  br label %exit
exit:
  call i1 @llvm.coro.end(ptr null, i1 false)
  ret void
}

; Verify that for both foo and bar, testval isn't put on the frame.
; CHECK: %foo.Frame = type { ptr, ptr, i1 }
; CHECK: %bar.Frame = type { ptr, ptr, i1 }

declare token @llvm.coro.id(i32, ptr readnone, ptr nocapture readonly, ptr)
declare i1 @llvm.coro.alloc(token) #3
declare i64 @llvm.coro.size.i64() #5
declare ptr @llvm.coro.begin(token, ptr writeonly) #3
declare token @llvm.coro.save(ptr) #3
declare ptr @llvm.coro.frame() #5
declare i8 @llvm.coro.suspend(token, i1) #3
declare ptr @llvm.coro.free(token, ptr nocapture readonly) #2
declare i1 @llvm.coro.end(ptr, i1) #3
declare void @llvm.lifetime.start.p0(i64, ptr nocapture) #4
declare void @llvm.lifetime.end.p0(i64, ptr nocapture) #4
