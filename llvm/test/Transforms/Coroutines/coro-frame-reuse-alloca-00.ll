; Check that we can handle spills of array allocas
; RUN: opt < %s -passes='cgscc(coro-split<reuse-storage>),simplifycfg,early-cse' -S | FileCheck %s

%struct.big_structure = type { [500 x i8] }
declare void @consume(ptr)

; Function Attrs: noinline optnone uwtable
define ptr @f(i1 %cond) presplitcoroutine {
entry:
  %data = alloca %struct.big_structure, align 1
  %data2 = alloca %struct.big_structure, align 1
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  br i1 %cond, label %then, label %else

then:
  call void @llvm.lifetime.start.p0(i64 500, ptr nonnull %data)
  call void @consume(ptr %data)
  %suspend.value = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %suspend.value, label %coro.ret [i8 0, label %resume
                                             i8 1, label %cleanup1]

resume:
  call void @llvm.lifetime.end.p0(i64 500, ptr nonnull %data)
  br label %cleanup1

cleanup1:
  call void @llvm.lifetime.end.p0(i64 500, ptr nonnull %data)
  br label %cleanup

else:
  call void @llvm.lifetime.start.p0(i64 500, ptr nonnull %data2)
  call void @consume(ptr %data2)
  %suspend.value2 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %suspend.value2, label %coro.ret [i8 0, label %resume2
                                              i8 1, label %cleanup2]

resume2:
  call void @llvm.lifetime.end.p0(i64 500, ptr nonnull %data2)
  br label %cleanup2

cleanup2:
  call void @llvm.lifetime.end.p0(i64 500, ptr nonnull %data2)
  br label %cleanup

cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %coro.ret
coro.ret:
  call i1 @llvm.coro.end(ptr %hdl, i1 0)
  ret ptr %hdl
}

; CHECK-LABEL: @f(
; CHECK: call ptr @malloc(i32 520)

declare ptr @llvm.coro.free(token, ptr)
declare i32 @llvm.coro.size.i32()
declare i8  @llvm.coro.suspend(token, i1)
declare void @llvm.coro.resume(ptr)
declare void @llvm.coro.destroy(ptr)

declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare i1 @llvm.coro.alloc(token)
declare ptr @llvm.coro.begin(token, ptr)
declare i1 @llvm.coro.end(ptr, i1)

declare noalias ptr @malloc(i32)
declare double @print(double)
declare void @free(ptr)

declare void @llvm.lifetime.start.p0(i64, ptr nocapture)
declare void @llvm.lifetime.end.p0(i64, ptr nocapture)
