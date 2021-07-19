; Tests that coro-split will optimize the lifetime.start maker of each local variable,
; sink them to the places after the suspend block.
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse,simplifycfg' -S | FileCheck %s

%"struct.std::coroutine_handle" = type { i8* }
%"struct.std::coroutine_handle.0" = type { %"struct.std::coroutine_handle" }
%"struct.lean_future<int>::Awaiter" = type { i32, %"struct.std::coroutine_handle.0" }

declare i8* @malloc(i64)
declare void @print(i32)

define void @a() "coroutine.presplit"="1" {
entry:
  %ref.tmp7 = alloca %"struct.lean_future<int>::Awaiter", align 8
  %testval = alloca i32
  %cast = bitcast i32* %testval to i8*
  ; lifetime of %testval starts here, but not used until await.ready.
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %cast)
  %id = call token @llvm.coro.id(i32 0, i8* null, i8* null, i8* null)
  %alloc = call i8* @malloc(i64 16) #3
  %vFrame = call noalias nonnull i8* @llvm.coro.begin(token %id, i8* %alloc)

  %save = call token @llvm.coro.save(i8* null)
  %Result.i19 = getelementptr inbounds %"struct.lean_future<int>::Awaiter", %"struct.lean_future<int>::Awaiter"* %ref.tmp7, i64 0, i32 0
  %suspend = call i8 @llvm.coro.suspend(token %save, i1 false)
  switch i8 %suspend, label %exit [
    i8 0, label %await.ready
    i8 1, label %exit
  ]
await.ready:
  %StrayCoroSave = call token @llvm.coro.save(i8* null)
  %val = load i32, i32* %Result.i19
  %test = load i32, i32* %testval
  call void @print(i32 %test)
  call void @llvm.lifetime.end.p0i8(i64 4, i8*  %cast)
  call void @print(i32 %val)
  br label %exit
exit:
  call i1 @llvm.coro.end(i8* null, i1 false)
  ret void
}

; CHECK-LABEL: @a.resume(
; CHECK:         %testval = alloca i32, align 4
; CHECK-NEXT:    getelementptr inbounds %a.Frame
; CHECK-NEXT:    %0 = bitcast i32* %testval to i8*
; CHECK-NEXT:    call void @llvm.lifetime.start.p0i8(i64 4, i8* %0)
; CHECK-NEXT:    getelementptr inbounds %"struct.lean_future<int>::Awaiter"
; CHECK-NEXT:    %val = load i32, i32* %Result
; CHECK-NEXT:    %test = load i32, i32* %testval
; CHECK-NEXT:    call void @print(i32 %test)
; CHECK-NEXT:    call void @llvm.lifetime.end.p0i8(i64 4, i8* %0)
; CHECK-NEXT:    call void @print(i32 %val)
; CHECK-NEXT:    ret void

declare token @llvm.coro.id(i32, i8* readnone, i8* nocapture readonly, i8*)
declare i1 @llvm.coro.alloc(token) #3
declare noalias nonnull i8* @"\01??2@YAPEAX_K@Z"(i64) local_unnamed_addr
declare i64 @llvm.coro.size.i64() #5
declare i8* @llvm.coro.begin(token, i8* writeonly) #3
declare void @"\01?puts@@YAXZZ"(...)
declare token @llvm.coro.save(i8*) #3
declare i8* @llvm.coro.frame() #5
declare i8 @llvm.coro.suspend(token, i1) #3
declare void @"\01??3@YAXPEAX@Z"(i8*) local_unnamed_addr #10
declare i8* @llvm.coro.free(token, i8* nocapture readonly) #2
declare i1 @llvm.coro.end(i8*, i1) #3
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #4
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #4
