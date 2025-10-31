; Check that we can handle spills of array allocas
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

declare void @consume.double.ptr(ptr)
declare void @consume.i32.ptr(ptr)

define ptr @f() presplitcoroutine {
entry:
  %prefix = alloca double
  %data = alloca i32, i32 4
  %suffix = alloca double
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  call void @consume.double.ptr(ptr %prefix)
  call void @consume.i32.ptr(ptr %data)
  call void @consume.double.ptr(ptr %suffix)
  %0 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %0, label %suspend [i8 0, label %resume
                                i8 1, label %cleanup]
resume:
  call void @consume.double.ptr(ptr %prefix)
  call void @consume.i32.ptr(ptr %data)
  call void @consume.double.ptr(ptr %suffix)
  br label %cleanup

cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend
suspend:
  call void @llvm.coro.end(ptr %hdl, i1 0, token none)
  ret ptr %hdl
}

; See if the array alloca was stored as an array field.
; CHECK-LABEL: %f.Frame = type { ptr, ptr, double, double, [4 x i32], i1 }

; See if we used correct index to access prefix, data, suffix (@f)
; CHECK-LABEL: @f(
; CHECK:       %[[PREFIX:.+]] = getelementptr inbounds %f.Frame, ptr %hdl, i32 0, i32 2
; CHECK-NEXT:  %[[DATA:.+]] = getelementptr inbounds %f.Frame, ptr %hdl, i32 0, i32 4
; CHECK-NEXT:  %[[SUFFIX:.+]] = getelementptr inbounds %f.Frame, ptr %hdl, i32 0, i32 3
; CHECK-NEXT:  call void @consume.double.ptr(ptr %[[PREFIX:.+]])
; CHECK-NEXT:  call void @consume.i32.ptr(ptr %[[DATA:.+]])
; CHECK-NEXT:  call void @consume.double.ptr(ptr %[[SUFFIX:.+]])
; CHECK: ret ptr

; See if we used correct index to access prefix, data, suffix (@f.resume)
; CHECK-LABEL: @f.resume(
; CHECK:       %[[PREFIX:.+]] = getelementptr inbounds %f.Frame, ptr %hdl, i32 0, i32 2
; CHECK:       %[[DATA:.+]] = getelementptr inbounds %f.Frame, ptr %hdl, i32 0, i32 4
; CHECK:       %[[SUFFIX:.+]] = getelementptr inbounds %f.Frame, ptr %hdl, i32 0, i32 3
; CHECK:       call void @consume.double.ptr(ptr %[[PREFIX]])
; CHECK-NEXT:  call void @consume.i32.ptr(ptr %[[DATA]])
; CHECK-NEXT:  call void @consume.double.ptr(ptr %[[SUFFIX]])

declare ptr @llvm.coro.free(token, ptr)
declare i32 @llvm.coro.size.i32()
declare i8  @llvm.coro.suspend(token, i1)
declare void @llvm.coro.resume(ptr)
declare void @llvm.coro.destroy(ptr)

declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare i1 @llvm.coro.alloc(token)
declare ptr @llvm.coro.begin(token, ptr)
declare void @llvm.coro.end(ptr, i1, token)

declare noalias ptr @malloc(i32)
declare double @print(double)
declare void @free(ptr)
