; Based on llvm/test/Transforms/Coroutines/coro-split-02.ll
; Corosplit will keep f1 and add 3 more functions.
; RUN: opt -passes='default<O1>,print<inline-advisor>' -training-log=/dev/null \
; RUN:   -S -enable-ml-inliner=development -keep-inline-advisor-for-printing < %s 2>&1 | FileCheck %s
; REQUIRES: have_tf_api
;
; CHECK: [MLInlineAdvisor] Nodes: 4 Edges: 0

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
  call void @llvm.lifetime.start.p0(i64 4, ptr %testval)
  %test = load i32, ptr %testval
  call void @print(i32 %test)
  call void @llvm.lifetime.end.p0(i64 4, ptr  %testval)
  call void @print(i32 %val)
  br label %exit
exit:
  call i1 @llvm.coro.end(ptr null, i1 false)
  ret void
}

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
