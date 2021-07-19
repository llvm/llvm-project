; Check that we can spills coro.begin from an inlined inner coroutine.
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

%g.Frame = type { void (%g.Frame*)*, void (%g.Frame*)*, i32, i1, i32 }

@g.resumers = private constant [3 x void (%g.Frame*)*] [void (%g.Frame*)* @g.dummy, void (%g.Frame*)* @g.dummy, void (%g.Frame*)* @g.dummy]

declare void @g.dummy(%g.Frame*)

define i8* @f() "coroutine.presplit"="1" {
entry:
  %id = call token @llvm.coro.id(i32 0, i8* null, i8* null, i8* null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call i8* @malloc(i32 %size)
  %hdl = call i8* @llvm.coro.begin(token %id, i8* %alloc)

  %innerid = call token @llvm.coro.id(i32 0, i8* null, i8* null, i8* bitcast ([3 x void (%g.Frame*)*]* @g.resumers to i8*))
  %innerhdl = call noalias nonnull i8* @llvm.coro.begin(token %innerid, i8* null)
  %gframe = bitcast i8* %innerhdl to %g.Frame*

  %tok = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %tok, label %suspend [i8 0, label %resume
                                i8 1, label %cleanup]
resume:
  %gvar.addr = getelementptr inbounds %g.Frame, %g.Frame* %gframe, i32 0, i32 4
  %gvar = load i32, i32* %gvar.addr
  call void @print.i32(i32 %gvar)
  br label %cleanup

cleanup:
  %mem = call i8* @llvm.coro.free(token %id, i8* %hdl)
  call void @free(i8* %mem)
  br label %suspend
suspend:
  call i1 @llvm.coro.end(i8* %hdl, i1 0)
  ret i8* %hdl
}

; See if the i8* for coro.begin was added to f.Frame
; CHECK-LABEL: %f.Frame = type { void (%f.Frame*)*, void (%f.Frame*)*, i8*, i1 }

; See if the g's coro.begin was spilled into the frame
; CHECK-LABEL: @f(
; CHECK: %innerid = call token @llvm.coro.id(i32 0, i8* null, i8* null, i8* bitcast ([3 x void (%g.Frame*)*]* @g.resumers to i8*))
; CHECK: %innerhdl = call noalias nonnull i8* @llvm.coro.begin(token %innerid, i8* null)
; CHECK: %[[spilladdr:.+]] = getelementptr inbounds %f.Frame, %f.Frame* %FramePtr, i32 0, i32 2
; CHECK: store i8* %innerhdl, i8** %[[spilladdr]]

; See if the coro.begin was loaded from the frame
; CHECK-LABEL: @f.resume(
; CHECK: %[[innerhdlAddr:.+]] = getelementptr inbounds %f.Frame, %f.Frame* %{{.+}}, i32 0, i32 2
; CHECK: %[[innerhdl:.+]] = load i8*, i8** %[[innerhdlAddr]]
; CHECK: %[[gframe:.+]] = bitcast i8* %[[innerhdl]] to %g.Frame*
; CHECK: %[[gvarAddr:.+]] = getelementptr inbounds %g.Frame, %g.Frame* %[[gframe]], i32 0, i32 4
; CHECK: %[[gvar:.+]] = load i32, i32* %[[gvarAddr]]
; CHECK: call void @print.i32(i32 %[[gvar]])

declare i8* @llvm.coro.free(token, i8*)
declare i32 @llvm.coro.size.i32()
declare i8  @llvm.coro.suspend(token, i1)

declare token @llvm.coro.id(i32, i8*, i8*, i8*)
declare i1 @llvm.coro.alloc(token)
declare i8* @llvm.coro.begin(token, i8*)
declare i1 @llvm.coro.end(i8*, i1)

declare noalias i8* @malloc(i32)
declare void @print.i32(i32)
declare void @free(i8*)
