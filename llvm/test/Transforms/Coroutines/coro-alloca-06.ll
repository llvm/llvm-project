; Test that in some simple cases allocas will not live on the frame even
; though their pointers are stored.
; RUN: opt < %s -passes='cgscc(coro-split),simplify-cfg,early-cse' -S | FileCheck %s

%handle = type { i8* }

define i8* @f() "coroutine.presplit"="1" {
entry:
  %0 = alloca %"handle", align 8
  %1 = alloca %"handle"*, align 8
  %id = call token @llvm.coro.id(i32 0, i8* null, i8* null, i8* null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call i8* @malloc(i32 %size)
  %hdl = call i8* @llvm.coro.begin(token %id, i8* %alloc)
  br label %tricky

tricky:
  %2 = call i8* @await_suspend()
  %3 = getelementptr inbounds %"handle", %"handle"* %0, i32 0, i32 0
  store i8* %2, i8** %3, align 8
  %4 = bitcast %"handle"** %1 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %4)
  store %"handle"* %0, %"handle"** %1, align 8
  %5 = load %"handle"*, %"handle"** %1, align 8
  %6 = getelementptr inbounds %"handle", %"handle"* %5, i32 0, i32 0
  %7 = load i8*, i8** %6, align 8
  %8 = bitcast %"handle"** %1 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %8)
  br label %finish

finish:
  %sp1 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp1, label %suspend [i8 0, label %resume
  i8 1, label %cleanup]
resume:
  br label %cleanup

cleanup:
  %mem = call i8* @llvm.coro.free(token %id, i8* %hdl)
  call void @free(i8* %mem)
  br label %suspend

suspend:
  call i1 @llvm.coro.end(i8* %hdl, i1 0)
  ret i8* %hdl
}

; CHECK:        %f.Frame = type { void (%f.Frame*)*, void (%f.Frame*)*, i1 }
; CHECK-LABEL: @f(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = alloca [[HANDLE:%.*]], align 8
; CHECK-NEXT:    [[TMP1:%.*]] = alloca %handle*, align 8

; CHECK:         [[TMP2:%.*]] = call i8* @await_suspend()
; CHECK-NEXT:    [[TMP3:%.*]] = getelementptr inbounds [[HANDLE]], %handle* [[TMP0]], i32 0, i32 0
; CHECK-NEXT:    store i8* [[TMP2]], i8** [[TMP3]], align 8
; CHECK-NEXT:    [[TMP4:%.*]] = bitcast %handle** [[TMP1]] to i8*
; CHECK-NEXT:    call void @llvm.lifetime.start.p0i8(i64 8, i8* [[TMP4]])
; CHECK-NEXT:    store %handle* [[TMP0]], %handle** [[TMP1]], align 8
; CHECK-NEXT:    call void @llvm.lifetime.end.p0i8(i64 8, i8* [[TMP4]])
;

declare i8* @llvm.coro.free(token, i8*)
declare i32 @llvm.coro.size.i32()
declare i8  @llvm.coro.suspend(token, i1)
declare void @llvm.coro.resume(i8*)
declare void @llvm.coro.destroy(i8*)

declare token @llvm.coro.id(i32, i8*, i8*, i8*)
declare i1 @llvm.coro.alloc(token)
declare i8* @llvm.coro.begin(token, i8*)
declare i1 @llvm.coro.end(i8*, i1)

declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture)
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture)

declare i8* @await_suspend()
declare void @print(i32* nocapture)
declare noalias i8* @malloc(i32)
declare void @free(i8*)
