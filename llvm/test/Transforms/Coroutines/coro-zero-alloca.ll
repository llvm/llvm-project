; RUN: opt < %s -O2 --enable-coroutines -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64--linux"

declare i8* @malloc(i64)
declare void @free(i8*)
declare void @usePointer(i8*)
declare void @usePointer2([0 x i8]*)

declare token @llvm.coro.id(i32, i8* readnone, i8* nocapture readonly, i8*)
declare i64 @llvm.coro.size.i64()
declare i8* @llvm.coro.begin(token, i8* writeonly)
declare i8 @llvm.coro.suspend(token, i1)
declare i1 @llvm.coro.end(i8*, i1)
declare i8* @llvm.coro.free(token, i8* nocapture readonly)
declare token @llvm.coro.save(i8*)

define void @foo() {
entry:
  %a0 = alloca [0 x i8]
  %a1 = alloca i32
  %a2 = alloca [0 x i8]
  %a3 = alloca [0 x i8]
  %a4 = alloca i16
  %a5 = alloca [0 x i8]
  %a0.cast = bitcast [0 x i8]* %a0 to i8*
  %a1.cast = bitcast i32* %a1 to i8*
  %a2.cast = bitcast [0 x i8]* %a2 to i8*
  %a3.cast = bitcast [0 x i8]* %a3 to i8*
  %a4.cast = bitcast i16* %a4 to i8*
  %coro.id = call token @llvm.coro.id(i32 0, i8* null, i8* null, i8* null)
  %coro.size = call i64 @llvm.coro.size.i64()
  %coro.alloc = call i8* @malloc(i64 %coro.size)
  %coro.state = call i8* @llvm.coro.begin(token %coro.id, i8* %coro.alloc)
  %coro.save = call token @llvm.coro.save(i8* %coro.state)
  %call.suspend = call i8 @llvm.coro.suspend(token %coro.save, i1 false)
  switch i8 %call.suspend, label %suspend [
    i8 0, label %wakeup
    i8 1, label %cleanup
  ]

wakeup:                                           ; preds = %entry
  call void @usePointer(i8* %a0.cast)
  call void @usePointer(i8* %a1.cast)
  call void @usePointer(i8* %a2.cast)
  call void @usePointer(i8* %a3.cast)
  call void @usePointer(i8* %a4.cast)
  call void @usePointer2([0 x i8]* %a5)
  br label %cleanup

suspend:                                          ; preds = %cleanup, %entry
  %unused = call i1 @llvm.coro.end(i8* %coro.state, i1 false)
  ret void

cleanup:                                          ; preds = %wakeup, %entry
  %coro.memFree = call i8* @llvm.coro.free(token %coro.id, i8* %coro.state)
  call void @free(i8* %coro.memFree)
  br label %suspend
}

; CHECK:       %foo.Frame = type { void (%foo.Frame*)*, void (%foo.Frame*)*, i32, i16, i1 }

; CHECK-LABEL: define internal fastcc void @foo.resume(%foo.Frame* noalias nonnull align 8 dereferenceable(24) %FramePtr) {
; CHECK-NEXT:    entry.resume:
; CHECK-NEXT:      %vFrame = bitcast %foo.Frame* %FramePtr to i8*
; CHECK-NEXT:      %a1.reload.addr = getelementptr inbounds %foo.Frame, %foo.Frame* %FramePtr, i64 0, i32 2
; CHECK-NEXT:      %a4.reload.addr = getelementptr inbounds %foo.Frame, %foo.Frame* %FramePtr, i64 0, i32 3
; CHECK-NEXT:      %a0.reload.addr = bitcast %foo.Frame* %FramePtr to [0 x i8]*
; CHECK-NEXT:      %a4.cast = bitcast i16* %a4.reload.addr to i8*
; CHECK-NEXT:      %a3.cast = bitcast %foo.Frame* %FramePtr to i8*
; CHECK-NEXT:      %a1.cast = bitcast i32* %a1.reload.addr to i8*
; CHECK-NEXT:      call void @usePointer(i8* nonnull %a3.cast)
; CHECK-NEXT:      call void @usePointer(i8* nonnull %a1.cast)
; CHECK-NEXT:      call void @usePointer(i8* nonnull %a3.cast)
; CHECK-NEXT:      call void @usePointer(i8* nonnull %a3.cast)
; CHECK-NEXT:      call void @usePointer(i8* nonnull %a4.cast)
; CHECK-NEXT:      call void @usePointer2([0 x i8]* nonnull %a0.reload.addr)
; CHECK-NEXT:      call void @free(i8* %vFrame)
; CHECK-NEXT:      ret void
; CHECK-NEXT:    }
