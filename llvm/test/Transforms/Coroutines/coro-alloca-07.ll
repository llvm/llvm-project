; Tests that CoroSplit can succesfully determine allocas should live on the frame
; if their aliases are used across suspension points through PHINode.
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

define ptr @f(i1 %n) presplitcoroutine {
entry:
  %x = alloca i64
  %y = alloca i64
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  br i1 %n, label %flag_true, label %flag_false

flag_true:
  call void @llvm.lifetime.start.p0(i64 8, ptr %x)
  br label %merge

flag_false:
  call void @llvm.lifetime.start.p0(i64 8, ptr %y)
  br label %merge

merge:
  %alias_phi = phi ptr [ %x, %flag_true ], [ %y, %flag_false ]
  store i8 1, ptr %alias_phi
  %sp1 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp1, label %suspend [i8 0, label %resume
                                  i8 1, label %cleanup]
resume:
  call void @print(ptr %alias_phi)
  br label %cleanup

cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend

suspend:
  call i1 @llvm.coro.end(ptr %hdl, i1 0, token none)
  ret ptr %hdl
}

declare ptr @llvm.coro.free(token, ptr)
declare i32 @llvm.coro.size.i32()
declare i8  @llvm.coro.suspend(token, i1)
declare void @llvm.coro.resume(ptr)
declare void @llvm.coro.destroy(ptr)

declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare i1 @llvm.coro.alloc(token)
declare ptr @llvm.coro.begin(token, ptr)
declare i1 @llvm.coro.end(ptr, i1, token)

declare void @llvm.lifetime.start.p0(i64, ptr nocapture)

declare void @print(ptr)
declare noalias ptr @malloc(i32)
declare void @free(ptr)

; Verify that both x and y are put in the frame.
; CHECK: %f.Frame = type { ptr, ptr, i64, i64, ptr, i1 }

; CHECK-LABEL: @f(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[ID:%.*]] = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr @f.resumers)
; CHECK-NEXT:    [[ALLOC:%.*]] = call ptr @malloc(i32 48)
; CHECK-NEXT:    [[HDL:%.*]] = call noalias nonnull ptr @llvm.coro.begin(token [[ID]], ptr [[ALLOC]])
; CHECK-NEXT:    store ptr @f.resume, ptr [[HDL]], align 8
; CHECK-NEXT:    [[DESTROY_ADDR:%.*]] = getelementptr inbounds [[F_FRAME:%f.Frame]], ptr [[HDL]], i32 0, i32 1
; CHECK-NEXT:    store ptr @f.destroy, ptr [[DESTROY_ADDR]], align 8
; CHECK-NEXT:    [[X_RELOAD_ADDR:%.*]] = getelementptr inbounds [[F_FRAME]], ptr [[HDL]], i32 0, i32 2
; CHECK-NEXT:    [[Y_RELOAD_ADDR:%.*]] = getelementptr inbounds [[F_FRAME]], ptr [[HDL]], i32 0, i32 3
; CHECK-NEXT:    br i1 [[N:%.*]], label [[MERGE:%.*]], label [[MERGE_FROM_FLAG_FALSE:%.*]]
; CHECK:       merge.from.flag_false:
; CHECK-NEXT:    br label [[MERGE:%.*]]
; CHECK:       merge:
; CHECK-NEXT:    [[ALIAS_PHI:%.*]] = phi ptr [ [[Y_RELOAD_ADDR]], [[MERGE_FROM_FLAG_FALSE]] ], [ [[X_RELOAD_ADDR]], [[ENTRY:%.*]] ]
; CHECK-NEXT:    [[ALIAS_PHI_SPILL_ADDR:%.*]] = getelementptr inbounds [[F_FRAME]], ptr [[HDL]], i32 0, i32 4
; CHECK-NEXT:    store ptr [[ALIAS_PHI]], ptr [[ALIAS_PHI_SPILL_ADDR]], align 8
; CHECK-NEXT:    store i8 1, ptr [[ALIAS_PHI]], align 1
; CHECK-NEXT:    [[INDEX_ADDR1:%.*]] = getelementptr inbounds [[F_FRAME]], ptr [[HDL]], i32 0, i32 5
; CHECK-NEXT:    store i1 false, ptr [[INDEX_ADDR1]], align 1
; CHECK-NEXT:    ret ptr [[HDL]]
;
;
; CHECK-LABEL: @f.resume(
; CHECK-NEXT:  entry.resume:
; CHECK-NEXT:    [[ALIAS_PHI_RELOAD_ADDR:%.*]] = getelementptr inbounds [[F_FRAME:%.*]], ptr [[HDL]], i32 0, i32 4
; CHECK-NEXT:    [[ALIAS_PHI_RELOAD:%.*]] = load ptr, ptr [[ALIAS_PHI_RELOAD_ADDR]], align 8
; CHECK-NEXT:    call void @print(ptr [[ALIAS_PHI_RELOAD]])
; CHECK-NEXT:    call void @free(ptr [[FRAMEPTR:%.*]])
; CHECK-NEXT:    ret void
