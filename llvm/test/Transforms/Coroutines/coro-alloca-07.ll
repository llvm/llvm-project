; Tests that CoroSplit can succesfully determine allocas should live on the frame
; if their aliases are used across suspension points through PHINode.
; RUN: opt < %s -passes='cgscc(coro-split),simplify-cfg,early-cse' -S | FileCheck %s

define i8* @f(i1 %n) "coroutine.presplit"="1" {
entry:
  %x = alloca i64
  %y = alloca i64
  %id = call token @llvm.coro.id(i32 0, i8* null, i8* null, i8* null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call i8* @malloc(i32 %size)
  %hdl = call i8* @llvm.coro.begin(token %id, i8* %alloc)
  br i1 %n, label %flag_true, label %flag_false

flag_true:
  %x.alias = bitcast i64* %x to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %x.alias)
  br label %merge

flag_false:
  %y.alias = bitcast i64* %y to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %y.alias)
  br label %merge

merge:
  %alias_phi = phi i8* [ %x.alias, %flag_true ], [ %y.alias, %flag_false ]
  store i8 1, i8* %alias_phi
  %sp1 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp1, label %suspend [i8 0, label %resume
                                  i8 1, label %cleanup]
resume:
  call void @print(i8* %alias_phi)
  br label %cleanup

cleanup:
  %mem = call i8* @llvm.coro.free(token %id, i8* %hdl)
  call void @free(i8* %mem)
  br label %suspend

suspend:
  call i1 @llvm.coro.end(i8* %hdl, i1 0)
  ret i8* %hdl
}

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

declare void @print(i8*)
declare noalias i8* @malloc(i32)
declare void @free(i8*)

; Verify that both x and y are put in the frame.
; CHECK: %f.Frame = type { void (%f.Frame*)*, void (%f.Frame*)*, i64, i64, i8*, i1 }

; CHECK-LABEL: @f(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[ID:%.*]] = call token @llvm.coro.id(i32 0, i8* null, i8* null, i8* bitcast ([3 x void (%f.Frame*)*]* @f.resumers to i8*))
; CHECK-NEXT:    [[ALLOC:%.*]] = call i8* @malloc(i32 48)
; CHECK-NEXT:    [[HDL:%.*]] = call noalias nonnull i8* @llvm.coro.begin(token [[ID]], i8* [[ALLOC]])
; CHECK-NEXT:    [[FRAMEPTR:%.*]] = bitcast i8* [[HDL]] to %f.Frame*
; CHECK-NEXT:    [[RESUME_ADDR:%.*]] = getelementptr inbounds [[F_FRAME:%.*]], %f.Frame* [[FRAMEPTR]], i32 0, i32 0
; CHECK-NEXT:    store void (%f.Frame*)* @f.resume, void (%f.Frame*)** [[RESUME_ADDR]], align 8
; CHECK-NEXT:    [[DESTROY_ADDR:%.*]] = getelementptr inbounds [[F_FRAME]], %f.Frame* [[FRAMEPTR]], i32 0, i32 1
; CHECK-NEXT:    store void (%f.Frame*)* @f.destroy, void (%f.Frame*)** [[DESTROY_ADDR]], align 8
; CHECK-NEXT:    [[X_RELOAD_ADDR:%.*]] = getelementptr inbounds [[F_FRAME]], %f.Frame* [[FRAMEPTR]], i32 0, i32 2
; CHECK-NEXT:    [[Y_RELOAD_ADDR:%.*]] = getelementptr inbounds [[F_FRAME]], %f.Frame* [[FRAMEPTR]], i32 0, i32 3
; CHECK-NEXT:    br i1 [[N:%.*]], label [[FLAG_TRUE:%.*]], label [[FLAG_FALSE:%.*]]
; CHECK:       flag_true:
; CHECK-NEXT:    [[X_ALIAS:%.*]] = bitcast i64* [[X_RELOAD_ADDR]] to i8*
; CHECK-NEXT:    call void @llvm.lifetime.start.p0i8(i64 8, i8* [[X_ALIAS]])
; CHECK-NEXT:    br label [[MERGE:%.*]]
; CHECK:       flag_false:
; CHECK-NEXT:    [[Y_ALIAS:%.*]] = bitcast i64* [[Y_RELOAD_ADDR]] to i8*
; CHECK-NEXT:    call void @llvm.lifetime.start.p0i8(i64 8, i8* [[Y_ALIAS]])
; CHECK-NEXT:    br label [[MERGE]]
; CHECK:       merge:
; CHECK-NEXT:    [[ALIAS_PHI:%.*]] = phi i8* [ [[X_ALIAS]], [[FLAG_TRUE]] ], [ [[Y_ALIAS]], [[FLAG_FALSE]] ]
; CHECK-NEXT:    [[ALIAS_PHI_SPILL_ADDR:%.*]] = getelementptr inbounds [[F_FRAME]], %f.Frame* [[FRAMEPTR]], i32 0, i32 4
; CHECK-NEXT:    store i8* [[ALIAS_PHI]], i8** [[ALIAS_PHI_SPILL_ADDR]], align 8
; CHECK-NEXT:    store i8 1, i8* [[ALIAS_PHI]], align 1
; CHECK-NEXT:    [[INDEX_ADDR1:%.*]] = getelementptr inbounds [[F_FRAME]], %f.Frame* [[FRAMEPTR]], i32 0, i32 5
; CHECK-NEXT:    store i1 false, i1* [[INDEX_ADDR1]], align 1
; CHECK-NEXT:    ret i8* [[HDL]]
;
;
; CHECK-LABEL: @f.resume(
; CHECK-NEXT:  entry.resume:
; CHECK-NEXT:    [[VFRAME:%.*]] = bitcast %f.Frame* [[FRAMEPTR:%.*]] to i8*
; CHECK-NEXT:    [[ALIAS_PHI_RELOAD_ADDR:%.*]] = getelementptr inbounds [[F_FRAME:%.*]], %f.Frame* [[FRAMEPTR]], i32 0, i32 4
; CHECK-NEXT:    [[ALIAS_PHI_RELOAD:%.*]] = load i8*, i8** [[ALIAS_PHI_RELOAD_ADDR]], align 8
; CHECK-NEXT:    call void @print(i8* [[ALIAS_PHI_RELOAD]])
; CHECK-NEXT:    call void @free(i8* [[VFRAME]])
; CHECK-NEXT:    ret void
