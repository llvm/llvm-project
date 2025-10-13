; Check that we will insert the correct padding if natural alignment of the
; spilled data does not match the alignment specified in alloca instruction.
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

%PackedStruct = type <{ i64 }>

declare void @consume(ptr)

define ptr @f() presplitcoroutine {
entry:
  %data = alloca %PackedStruct, align 32
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  call void @consume(ptr %data)
  %0 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %0, label %suspend [i8 0, label %resume
                                i8 1, label %cleanup]
resume:
  call void @consume(ptr %data)
  br label %cleanup

cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend
suspend:
  call void @llvm.coro.end(ptr %hdl, i1 0, token none)
  ret ptr %hdl
}

; See if the padding was inserted before PackedStruct
; CHECK-LABEL: %f.Frame = type { ptr, ptr, i1, [15 x i8], %PackedStruct }

; See if we used correct index to access packed struct (padding is field 3)
; CHECK-LABEL: @f(
; CHECK:       %[[DATA:.+]] = getelementptr inbounds %f.Frame, ptr %hdl, i32 0, i32 4
; CHECK-NEXT:  call void @consume(ptr %[[DATA]])
; CHECK: ret ptr

; See if we used correct index to access packed struct (padding is field 3)
; CHECK-LABEL: @f.resume(
; CHECK:       %[[DATA:.+]] = getelementptr inbounds %f.Frame, ptr %hdl, i32 0, i32 4
; CHECK-NEXT:  call void @consume(ptr %[[DATA]])
; CHECK: ret void

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
