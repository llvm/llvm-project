; Tests that allocas escaped through function calls will live on the frame.
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

define i8* @f() presplitcoroutine {
entry:
  %x = alloca i64
  %y = alloca i64
  %id = call token @llvm.coro.id(i32 0, i8* null, i8* null, i8* null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call i8* @malloc(i32 %size)
  %hdl = call i8* @llvm.coro.begin(token %id, i8* %alloc)
  %x.alias = bitcast i64* %x to i32*
  call void @capture_call(i32* %x.alias)
  %y.alias = bitcast i64* %y to i32*
  call void @nocapture_call(i32* %y.alias)
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

; %x needs to go to the frame since it's escaped; %y will stay as local since it doesn't escape.
; CHECK:        %f.Frame = type { void (%f.Frame*)*, void (%f.Frame*)*, i64, i1 }
; CHECK-LABEL:  define i8* @f()
; CHECK:          %y = alloca i64, align 8
; CHECK:          %x.reload.addr = getelementptr inbounds %f.Frame, %f.Frame* %FramePtr, i32 0, i32 2
; CHECK:          %x.alias = bitcast i64* %x.reload.addr to i32*
; CHECK:          call void @capture_call(i32* %x.alias)
; CHECK:          %y.alias = bitcast i64* %y to i32*
; CHECK:          call void @nocapture_call(i32* %y.alias)

declare i8* @llvm.coro.free(token, i8*)
declare i32 @llvm.coro.size.i32()
declare i8  @llvm.coro.suspend(token, i1)
declare void @llvm.coro.resume(i8*)
declare void @llvm.coro.destroy(i8*)

declare token @llvm.coro.id(i32, i8*, i8*, i8*)
declare i1 @llvm.coro.alloc(token)
declare i8* @llvm.coro.begin(token, i8*)
declare i1 @llvm.coro.end(i8*, i1)

declare void @capture_call(i32*)
declare void @nocapture_call(i32* nocapture)
declare noalias i8* @malloc(i32)
declare void @free(i8*)
