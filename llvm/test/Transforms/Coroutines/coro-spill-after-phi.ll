; Verifies that we insert spills of PHI instruction _after) all PHI Nodes
; RUN: opt < %s -passes=coro-split -S | FileCheck %s

define i8* @f(i1 %n) "coroutine.presplit"="1" {
entry:
  %id = call token @llvm.coro.id(i32 0, i8* null, i8* null, i8* null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call i8* @malloc(i32 %size)
  %hdl = call i8* @llvm.coro.begin(token %id, i8* %alloc)
  br i1 %n, label %begin, label %alt
alt:
  br label %begin

begin:
  %phi1 = phi i32 [ 0, %entry ], [ 2, %alt ]
  %phi2 = phi i32 [ 1, %entry ], [ 3, %alt ]

  %sp1 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp1, label %suspend [i8 0, label %resume
                                  i8 1, label %cleanup]
resume:
  call i32 @print(i32 %phi1)
  call i32 @print(i32 %phi2)
  br label %cleanup

cleanup:
  %mem = call i8* @llvm.coro.free(token %id, i8* %hdl)
  call void @free(i8* %mem)
  br label %suspend
suspend:
  call i1 @llvm.coro.end(i8* %hdl, i1 0)
  ret i8* %hdl
}

; Verifies that the both phis are stored correctly in the coroutine frame
; CHECK: %f.Frame = type { void (%f.Frame*)*, void (%f.Frame*)*, i32, i32, i1 }
; CHECK-LABEL: @f(
; CHECK: store void (%f.Frame*)* @f.destroy, void (%f.Frame*)** %destroy.addr
; CHECK: %phi1 = select i1 %n, i32 0, i32 2
; CHECK: %phi2 = select i1 %n, i32 1, i32 3
; CHECK: %phi2.spill.addr = getelementptr inbounds %f.Frame, %f.Frame* %FramePtr, i32 0, i32 3
; CHECK: store i32 %phi2, i32* %phi2.spill.addr
; CHECK: %phi1.spill.addr = getelementptr inbounds %f.Frame, %f.Frame* %FramePtr, i32 0, i32 2
; CHECK: store i32 %phi1, i32* %phi1.spill.addr
; CHECK: ret i8* %hdl

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
declare i32 @print(i32)
declare void @free(i8*)
