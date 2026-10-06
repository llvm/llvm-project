; RUN: opt %s -passes='cgscc(coro-split)' -S | FileCheck %s

; Verify that values are only rematerializes across a suspend when doing so
; introduces no new spill.

; %v's operands are two call results that are otherwise dead across the suspend.
; Rematerializing %v would force both calls' results into the frame, so spill %v.

; CHECK-LABEL: @f_nonfree(
; CHECK: %[[V:.*]] = add i32 %[[C1:.*]], %[[C2:.*]]
; CHECK: store i32 %[[V]]
; CHECK-NOT: store i32 %[[C1]]
; CHECK-NOT: store i32 %[[C2]]
define ptr @f_nonfree() presplitcoroutine {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr @f_nonfree, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  %c1 = call i32 @opaque()
  %c2 = call i32 @opaque()
  %v = add i32 %c1, %c2
  %sp1 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp1, label %suspend [i8 0, label %resume
                                  i8 1, label %cleanup]
resume:
  call void @print(i32 %v)
  br label %cleanup
cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend
suspend:
  call void @llvm.coro.end(ptr %hdl, i1 0, token none)
  ret ptr %hdl
}

; %inc's only non-constant operand is an argument -> free -> still rematerialized
; with the flag on: %n is spilled, %inc is not.

; CHECK-LABEL: @f_free(
; CHECK-SAME: %[[N:[[:alnum:]]+]]
; CHECK: store i32 %[[N]]
; CHECK-NOT: store i32
; CHECK: switch
define ptr @f_free(i32 %n) presplitcoroutine {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr @f_free, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  %inc = add i32 %n, 1
  %sp1 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp1, label %suspend [i8 0, label %resume
                                  i8 1, label %cleanup]
resume:
  call void @print(i32 %inc)
  br label %cleanup
cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend
suspend:
  call void @llvm.coro.end(ptr %hdl, i1 0, token none)
  ret ptr %hdl
}

; %v is materializable and still rematerialized: its operand %m crosses the
; suspend on its own (used directly in resume), so %m is "free as an operand,
; i.e., its frame slot is referenced at no new cost. But %m must NOT itself be
; rematerialized, because %m's leaf %leaf is a call result that is otherwise
; dead across the suspend.

; CHECK-LABEL: @f_nested(
; CHECK: %[[M:.*]] = add i32 %[[LEAF:.*]], 1
; CHECK-NOT: store i32 %[[LEAF]]
; CHECK: store i32 %[[M]]
; CHECK-NOT: store i32 %[[LEAF]]
define ptr @f_nested() presplitcoroutine {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr @f_nested, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  %leaf = call i32 @opaque()
  %m = add i32 %leaf, 1
  %v = add i32 %m, 2
  %sp1 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp1, label %suspend [i8 0, label %resume
                                  i8 1, label %cleanup]
resume:
  call void @print(i32 %v)
  call void @print(i32 %m)
  br label %cleanup
cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend
suspend:
  call void @llvm.coro.end(ptr %hdl, i1 0, token none)
  ret ptr %hdl
}

; %res is not rematerialized, as it would force both %c1 and %c2 into the
; coroutine frame instead of %res.

; CHECK-LABEL: @f_two_args(
; CHECK: %[[C1:.*]] = call i32 @opaque
; CHECK: %[[C2:.*]] = call i32 @opaque
; CHECK: %[[RES:.*]] = mul i32 %[[C1]], %[[C2]]
; CHECK: store i32 %[[RES]]
define ptr @f_two_args() presplitcoroutine {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr @f_two_args, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  %c1 = call i32 @opaque()
  %c2 = call i32 @opaque()
  %res = mul i32 %c1, %c2
  %sp1 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp1, label %suspend [i8 0, label %resume
                                  i8 1, label %cleanup]
resume:
  call void @print(i32 %res)
  br label %cleanup
cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend
suspend:
  call void @llvm.coro.end(ptr %hdl, i1 0, token none)
  ret ptr %hdl
}

; coro-split emits all ramp functions before any of the resume/destroy/cleanup
; thunks, so these checks (which must scan forward) come after all defines above.

; CHECK-LABEL: @f_nonfree.resume(
; CHECK: %[[V_RELOAD:.*]] = load i32
; CHECK-NOT: add i32
; CHECK: call void @print(i32 %[[V_RELOAD]])

; CHECK-LABEL: @f_free.resume(
; CHECK: %[[N_RELOAD:.*]] = load i32
; CHECK: add i32 %[[N_RELOAD]], 1

; CHECK-LABEL: @f_nested.resume(
; CHECK: %[[M_RELOAD:.*]] = load i32
; CHECK-NOT: add i32 %{{.*}}, 1
; CHECK: add i32 %[[M_RELOAD]], 2
; CHECK-NOT: add i32 %{{.*}}, 1

; CHECK-LABEL: @f_two_args.resume(
; CHECK: %[[RES_RELOAD:.*]] = load i32
; CHECK-NOT: mul i32
; CHECK: call void @print(i32 %[[RES_RELOAD]])

declare ptr @llvm.coro.free(token, ptr)
declare i32 @llvm.coro.size.i32()
declare i8  @llvm.coro.suspend(token, i1)
declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare ptr @llvm.coro.begin(token, ptr)
declare void @llvm.coro.end(ptr, i1, token)
declare noalias ptr @malloc(i32)
declare i32 @opaque()
declare void @print(i32)
declare void @free(ptr)
