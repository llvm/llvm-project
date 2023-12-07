; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

%"struct.std::coroutine_handle" = type { ptr }
%"struct.std::coroutine_handle.0" = type { %"struct.std::coroutine_handle" }
%"struct.lean_future<int>::Awaiter" = type { i32, %"struct.std::coroutine_handle.0" }

declare ptr @malloc(i64)

%i8.array = type { [100 x i8] }
declare void @consume.i8(ptr)

; The testval lives across suspend point so that it should be put on the frame.
; However, part of testval has lifetime marker which indicates the part
; wouldn't live across suspend point.
; This test whether or not %testval would be put on the frame by ignoring the
; partial lifetime markers.
define void @foo(ptr %to_store) presplitcoroutine {
entry:
  %testval = alloca %i8.array
  %subrange = getelementptr inbounds %i8.array, ptr %testval, i64 0, i32 0, i64 50
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %alloc = call ptr @malloc(i64 16) #3
  %vFrame = call noalias nonnull ptr @llvm.coro.begin(token %id, ptr %alloc)

  call void @llvm.lifetime.start.p0(i64 50, ptr %subrange)
  call void @consume.i8(ptr %subrange)
  call void @llvm.lifetime.end.p0(i64 50, ptr  %subrange)
  store ptr %testval, ptr %to_store

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

; Verify that for both foo and bar, testval isn't put on the frame.
; CHECK: %foo.Frame = type { ptr, ptr, %i8.array, i1 }

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
