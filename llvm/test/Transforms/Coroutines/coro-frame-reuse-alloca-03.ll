; Check that we should not reuse alloca sotrage in O0.
; RUN: opt < %s -passes=coro-split -S | FileCheck %s

%struct.big_structure = type { [500 x i8] }
declare void @consume(%struct.big_structure*)

; Function Attrs: noinline optnone uwtable
define i8* @f(i1 %cond) "coroutine.presplit"="1" {
entry:
  %data = alloca %struct.big_structure, align 1
  %data2 = alloca %struct.big_structure, align 1
  %id = call token @llvm.coro.id(i32 0, i8* null, i8* null, i8* null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call i8* @malloc(i32 %size)
  %hdl = call i8* @llvm.coro.begin(token %id, i8* %alloc)
  br i1 %cond, label %then, label %else

then:
  %0 = bitcast %struct.big_structure* %data to i8*
  call void @llvm.lifetime.start.p0i8(i64 500, i8* nonnull %0)
  call void @consume(%struct.big_structure* %data)
  %suspend.value = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %suspend.value, label %coro.ret [i8 0, label %resume
                                             i8 1, label %cleanup1]

resume:
  call void @llvm.lifetime.end.p0i8(i64 500, i8* nonnull %0)
  br label %cleanup1

cleanup1:
  call void @llvm.lifetime.end.p0i8(i64 500, i8* nonnull %0)
  br label %cleanup

else:
  %1 = bitcast %struct.big_structure* %data2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 500, i8* nonnull %1)
  call void @consume(%struct.big_structure* %data2)
  %suspend.value2 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %suspend.value2, label %coro.ret [i8 0, label %resume2
                                              i8 1, label %cleanup2]

resume2:
  call void @llvm.lifetime.end.p0i8(i64 500, i8* nonnull %1)
  br label %cleanup2

cleanup2:
  call void @llvm.lifetime.end.p0i8(i64 500, i8* nonnull %1)
  br label %cleanup

cleanup:
  %mem = call i8* @llvm.coro.free(token %id, i8* %hdl)
  call void @free(i8* %mem)
  br label %coro.ret
coro.ret:
  call i1 @llvm.coro.end(i8* %hdl, i1 0)
  ret i8* %hdl
}

; CHECK-LABEL: @f(
; CHECK: call i8* @malloc(i32 1024)

declare i8* @llvm.coro.free(token, i8*)
declare i32 @llvm.coro.size.i32()
declare i8  @llvm.coro.suspend(token, i1)
declare void @llvm.coro.resume(i8*)
declare void @llvm.coro.destroy(i8*)

declare token @llvm.coro.id(i32, i8*, i8*, i8*)
declare i1 @llvm.coro.alloc(token)
declare i8* @llvm.coro.begin(token, i8*)
declare i1 @llvm.coro.end(i8*, i1)

declare noalias i8* @malloc(i32)
declare double @print(double)
declare void @free(i8*)

declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture)
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture)
