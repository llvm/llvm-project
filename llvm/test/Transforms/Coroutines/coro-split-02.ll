; Tests that coro-split can handle the case when a code after coro.suspend uses
; a value produces between coro.save and coro.suspend (%Result.i19)
; and checks whether stray coro.saves are properly removed
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse,simplifycfg' -S | FileCheck %s

%"struct.std::coroutine_handle" = type { ptr }
%"struct.std::coroutine_handle.0" = type { %"struct.std::coroutine_handle" }
%"struct.lean_future<int>::Awaiter" = type { i32, %"struct.std::coroutine_handle.0" }

declare ptr @malloc(i64)
declare void @print(i32)

define void @a() presplitcoroutine {
entry:
  %ref.tmp7 = alloca %"struct.lean_future<int>::Awaiter", align 8
  %testval = alloca i32
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
  %val = load i32, ptr %ref.tmp7
  call void @llvm.lifetime.start.p0(ptr %testval)
  %test = load i32, ptr %testval
  call void @print(i32 %test)
  call void @llvm.lifetime.end.p0(ptr  %testval)
  call void @print(i32 %val)
  br label %exit
exit:
  call i1 @llvm.coro.end(ptr null, i1 false, token none)
  ret void
}

; CHECK-LABEL: @a.resume(
; CHECK:         %testval = alloca i32
; CHECK-NOT:     call token @llvm.coro.save(ptr null)
; CHECK:         %val = load i32, ptr %ref.tmp7
; CHECK-NEXT:    call void @llvm.lifetime.start.p0(ptr %testval)
; CHECK-NEXT:    %test = load i32, ptr %testval
; CHECK-NEXT:    call void @print(i32 %test)
; CHECK-NEXT:    call void @llvm.lifetime.end.p0(ptr %testval)
; CHECK-NEXT:    call void @print(i32 %val)
; CHECK-NEXT:    ret void

declare token @llvm.coro.id(i32, ptr readnone, ptr nocapture readonly, ptr)
declare i1 @llvm.coro.alloc(token) #3
declare noalias nonnull ptr @"\01??2@YAPEAX_K@Z"(i64) local_unnamed_addr
declare i64 @llvm.coro.size.i64() #5
declare ptr @llvm.coro.begin(token, ptr writeonly) #3
declare void @"\01?puts@@YAXZZ"(...)
declare token @llvm.coro.save(ptr) #3
declare ptr @llvm.coro.frame() #5
declare i8 @llvm.coro.suspend(token, i1) #3
declare void @"\01??3@YAXPEAX@Z"(ptr) local_unnamed_addr #10
declare ptr @llvm.coro.free(token, ptr nocapture readonly) #2
declare i1 @llvm.coro.end(ptr, i1, token) #3
declare void @llvm.lifetime.start.p0(ptr nocapture) #4
declare void @llvm.lifetime.end.p0(ptr nocapture) #4
