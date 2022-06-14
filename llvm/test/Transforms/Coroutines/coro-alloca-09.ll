; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

%"struct.std::coroutine_handle" = type { i8* }
%"struct.std::coroutine_handle.0" = type { %"struct.std::coroutine_handle" }
%"struct.lean_future<int>::Awaiter" = type { i32, %"struct.std::coroutine_handle.0" }

declare i8* @malloc(i64)

%i8.array = type { [100 x i8] }
declare void @consume.i8(i8*)

; The testval lives across suspend point so that it should be put on the frame.
; However, part of testval has lifetime marker which indicates the part
; wouldn't live across suspend point.
; This test whether or not %testval would be put on the frame by ignoring the
; partial lifetime markers.
define void @foo(%i8.array** %to_store) presplitcoroutine {
entry:
  %testval = alloca %i8.array
  %subrange = getelementptr inbounds %i8.array, %i8.array* %testval, i64 0, i32 0, i64 50
  %id = call token @llvm.coro.id(i32 0, i8* null, i8* null, i8* null)
  %alloc = call i8* @malloc(i64 16) #3
  %vFrame = call noalias nonnull i8* @llvm.coro.begin(token %id, i8* %alloc)

  call void @llvm.lifetime.start.p0i8(i64 50, i8* %subrange)
  call void @consume.i8(i8* %subrange)
  call void @llvm.lifetime.end.p0i8(i64 50, i8*  %subrange)
  store %i8.array* %testval, %i8.array** %to_store

  %save = call token @llvm.coro.save(i8* null)
  %suspend = call i8 @llvm.coro.suspend(token %save, i1 false)
  switch i8 %suspend, label %exit [
    i8 0, label %await.ready
    i8 1, label %exit
  ]
await.ready:
  %StrayCoroSave = call token @llvm.coro.save(i8* null)
  br label %exit
exit:
  call i1 @llvm.coro.end(i8* null, i1 false)
  ret void
}

; Verify that for both foo and bar, testval isn't put on the frame.
; CHECK: %foo.Frame = type { void (%foo.Frame*)*, void (%foo.Frame*)*, %i8.array, i1 }

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
