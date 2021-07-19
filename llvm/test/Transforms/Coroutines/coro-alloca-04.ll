; Tests that CoroSplit can succesfully determine allocas should live on the frame
; if their aliases are used across suspension points through PHINode.
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

define i8* @f(i1 %n) "coroutine.presplit"="1" {
entry:
  %x = alloca i64
  br i1 %n, label %flag_true, label %flag_false

flag_true:
  %x.alias1 = bitcast i64* %x to i32*
  br label %merge

flag_false:
  %x.alias2 = bitcast i64* %x to i32*
  br label %merge

merge:
  %alias_phi = phi i32* [ %x.alias1, %flag_true ], [ %x.alias2, %flag_false ]
  %id = call token @llvm.coro.id(i32 0, i8* null, i8* null, i8* null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call i8* @malloc(i32 %size)
  %hdl = call i8* @llvm.coro.begin(token %id, i8* %alloc)
  %sp1 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp1, label %suspend [i8 0, label %resume
                                  i8 1, label %cleanup]
resume:
  call void @print(i32* %alias_phi)
  br label %cleanup

cleanup:
  %mem = call i8* @llvm.coro.free(token %id, i8* %hdl)
  call void @free(i8* %mem)
  br label %suspend

suspend:
  call i1 @llvm.coro.end(i8* %hdl, i1 0)
  ret i8* %hdl
}

; both %x and %alias_phi would go to the frame.
; CHECK:       %f.Frame = type { void (%f.Frame*)*, void (%f.Frame*)*, i64, i32*, i1 }
; CHECK-LABEL: @f(
; CHECK:         store void (%f.Frame*)* @f.destroy, void (%f.Frame*)** %destroy.addr
; CHECK-NEXT:    %0 = getelementptr inbounds %f.Frame, %f.Frame* %FramePtr, i32 0, i32 2
; CHECK-NEXT:    %1 = bitcast i64* %0 to i8*
; CHECK-NEXT:    %2 = bitcast i8* %1 to i32*
; CHECK:         %alias_phi.spill.addr = getelementptr inbounds %f.Frame, %f.Frame* %FramePtr, i32 0, i32 3
; CHECK-NEXT:    store i32* %2, i32** %alias_phi.spill.addr

declare i8* @llvm.coro.free(token, i8*)
declare i32 @llvm.coro.size.i32()
declare i8  @llvm.coro.suspend(token, i1)
declare void @llvm.coro.resume(i8*)
declare void @llvm.coro.destroy(i8*)

declare token @llvm.coro.id(i32, i8*, i8*, i8*)
declare i1 @llvm.coro.alloc(token)
declare i8* @llvm.coro.begin(token, i8*)
declare i1 @llvm.coro.end(i8*, i1)

declare void @print(i32*)
declare noalias i8* @malloc(i32)
declare void @free(i8*)
