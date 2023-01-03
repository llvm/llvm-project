; Tests that coro-split will optimize the lifetime.start maker of each local variable,
; sink them to the places after the suspend block.
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

%"struct.std::coroutine_handle" = type { ptr }
%"struct.std::coroutine_handle.0" = type { %"struct.std::coroutine_handle" }
%"struct.lean_future<int>::Awaiter" = type { i32, %"struct.std::coroutine_handle.0" }

declare i1 @getcond()
declare ptr @malloc(i64)
declare void @print(i32)

define void @a() presplitcoroutine {
entry:
  %ref.tmp7 = alloca %"struct.lean_future<int>::Awaiter", align 8
  %testval = alloca i32
  ; lifetime of %testval starts here, but not used until await.ready.
  call void @llvm.lifetime.start.p0(i64 4, ptr %testval)
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %alloc = call ptr @malloc(i64 16) #3
  %vFrame = call noalias nonnull ptr @llvm.coro.begin(token %id, ptr %alloc)
  %testcond = call i1 @getcond()
  br i1 %testcond, label %if.suspend, label %else.direct

if.suspend:
  %save = call token @llvm.coro.save(ptr null)
  %suspend = call i8 @llvm.coro.suspend(token %save, i1 false)
  switch i8 %suspend, label %exit [
    i8 0, label %await.ready
    i8 1, label %exit
  ]

else.direct:
  br label %after.await

await.ready:
  %StrayCoroSave = call token @llvm.coro.save(ptr null)
  %val = load i32, ptr %ref.tmp7
  %test = load i32, ptr %testval
  call void @print(i32 %test)
  call void @print(i32 %val)
  br label %after.await

after.await:
  %test1 = load i32, ptr %testval
  call void @print(i32 %test1)
  call void @llvm.lifetime.end.p0(i64 4, ptr  %testval)
  br label %exit

exit:
  call i1 @llvm.coro.end(ptr null, i1 false)
  ret void
}

; CHECK-LABEL: @a.resume(
; CHECK:    %[[VAL:testval.+]] = getelementptr inbounds %a.Frame
; CHECK-NOT:     call void @llvm.lifetime.start.p0(i64 4, ptr %{{.*}})
; CHECK:         %test = load i32, ptr %[[VAL]]

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
declare i1 @llvm.coro.end(ptr, i1) #3
declare void @llvm.lifetime.start.p0(i64, ptr nocapture) #4
declare void @llvm.lifetime.end.p0(i64, ptr nocapture) #4
