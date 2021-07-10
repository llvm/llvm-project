; Check that we will insert the correct padding if natural alignment of the
; spilled data does not match the alignment specified in alloca instruction.
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

%PackedStruct = type <{ i64 }>

declare void @consume(%PackedStruct*)

define i8* @f() "coroutine.presplit"="1" {
entry:
  %data = alloca %PackedStruct, align 32
  %id = call token @llvm.coro.id(i32 0, i8* null, i8* null, i8* null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call i8* @malloc(i32 %size)
  %hdl = call i8* @llvm.coro.begin(token %id, i8* %alloc)
  call void @consume(%PackedStruct* %data)
  %0 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %0, label %suspend [i8 0, label %resume
                                i8 1, label %cleanup]
resume:
  call void @consume(%PackedStruct* %data)
  br label %cleanup

cleanup:
  %mem = call i8* @llvm.coro.free(token %id, i8* %hdl)
  call void @free(i8* %mem)
  br label %suspend
suspend:
  call i1 @llvm.coro.end(i8* %hdl, i1 0)
  ret i8* %hdl
}

; See if the padding was inserted before PackedStruct
; CHECK-LABEL: %f.Frame = type { void (%f.Frame*)*, void (%f.Frame*)*, i1, [15 x i8], %PackedStruct }

; See if we used correct index to access packed struct (padding is field 3)
; CHECK-LABEL: @f(
; CHECK:       %[[DATA:.+]] = getelementptr inbounds %f.Frame, %f.Frame* %FramePtr, i32 0, i32 4
; CHECK-NEXT:  call void @consume(%PackedStruct* %[[DATA]])
; CHECK: ret i8*

; See if we used correct index to access packed struct (padding is field 3)
; CHECK-LABEL: @f.resume(
; CHECK:       %[[DATA:.+]] = getelementptr inbounds %f.Frame, %f.Frame* %FramePtr, i32 0, i32 4
; CHECK-NEXT:  call void @consume(%PackedStruct* %[[DATA]])
; CHECK: ret void

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
