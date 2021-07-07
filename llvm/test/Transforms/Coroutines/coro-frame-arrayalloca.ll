; Check that we can handle spills of array allocas
; RUN: opt < %s -passes='cgscc(coro-split),simplify-cfg,early-cse' -S | FileCheck %s

declare void @consume.double.ptr(double*)
declare void @consume.i32.ptr(i32*)

define i8* @f() "coroutine.presplit"="1" {
entry:
  %prefix = alloca double
  %data = alloca i32, i32 4
  %suffix = alloca double
  %id = call token @llvm.coro.id(i32 0, i8* null, i8* null, i8* null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call i8* @malloc(i32 %size)
  %hdl = call i8* @llvm.coro.begin(token %id, i8* %alloc)
  call void @consume.double.ptr(double* %prefix)
  call void @consume.i32.ptr(i32* %data)
  call void @consume.double.ptr(double* %suffix)
  %0 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %0, label %suspend [i8 0, label %resume
                                i8 1, label %cleanup]
resume:
  call void @consume.double.ptr(double* %prefix)
  call void @consume.i32.ptr(i32* %data)
  call void @consume.double.ptr(double* %suffix)
  br label %cleanup

cleanup:
  %mem = call i8* @llvm.coro.free(token %id, i8* %hdl)
  call void @free(i8* %mem)
  br label %suspend
suspend:
  call i1 @llvm.coro.end(i8* %hdl, i1 0)
  ret i8* %hdl
}

; See if the array alloca was stored as an array field.
; CHECK-LABEL: %f.Frame = type { void (%f.Frame*)*, void (%f.Frame*)*, double, double, [4 x i32], i1 }

; See if we used correct index to access prefix, data, suffix (@f)
; CHECK-LABEL: @f(
; CHECK:       %[[PREFIX:.+]] = getelementptr inbounds %f.Frame, %f.Frame* %FramePtr, i32 0, i32 2
; CHECK-NEXT:  %[[DATA:.+]] = getelementptr inbounds %f.Frame, %f.Frame* %FramePtr, i32 0, i32 4
; CHECK-NEXT:  %[[SUFFIX:.+]] = getelementptr inbounds %f.Frame, %f.Frame* %FramePtr, i32 0, i32 3
; CHECK-NEXT:  call void @consume.double.ptr(double* %[[PREFIX:.+]])
; CHECK-NEXT:  call void @consume.i32.ptr(i32* %[[DATA:.+]])
; CHECK-NEXT:  call void @consume.double.ptr(double* %[[SUFFIX:.+]])
; CHECK: ret i8*

; See if we used correct index to access prefix, data, suffix (@f.resume)
; CHECK-LABEL: @f.resume(
; CHECK:       %[[PREFIX:.+]] = getelementptr inbounds %f.Frame, %f.Frame* %FramePtr, i32 0, i32 2
; CHECK:       %[[DATA:.+]] = getelementptr inbounds %f.Frame, %f.Frame* %FramePtr, i32 0, i32 4
; CHECK:       %[[SUFFIX:.+]] = getelementptr inbounds %f.Frame, %f.Frame* %FramePtr, i32 0, i32 3
; CHECK:       call void @consume.double.ptr(double* %[[PREFIX]])
; CHECK-NEXT:  call void @consume.i32.ptr(i32* %[[DATA]])
; CHECK-NEXT:  call void @consume.double.ptr(double* %[[SUFFIX]])

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
