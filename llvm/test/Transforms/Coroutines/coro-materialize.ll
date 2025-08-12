; Verifies that we materialize instruction across suspend points
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

; See that we only spilled one value for f
; CHECK: %f.Frame = type { ptr, ptr, i32, i1 }
; CHECK: %f_optnone.Frame = type { ptr, ptr, i32, i32, i1 }
; Check other variants where different levels of materialization are achieved
; CHECK: %f_multiple_remat.Frame = type { ptr, ptr, i32, i1 }
; CHECK: %f_common_def.Frame = type { ptr, ptr, i32, i1 }
; CHECK: %f_common_def_multi_result.Frame = type { ptr, ptr, i32, i1 }
; CHECK-LABEL: @f(
; CHECK-LABEL: @f_optnone
; CHECK-LABEL: @f_multiple_remat(
; CHECK-LABEL: @f_common_def(
; CHECK-LABEL: @f_common_def_multi_result(

define ptr @f(i32 %n) presplitcoroutine {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)

  %inc1 = add i32 %n, 1
  %sp1 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp1, label %suspend [i8 0, label %resume1
                                  i8 1, label %cleanup]
resume1:
  %inc2 = add i32 %inc1, 1
  %sp2 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp2, label %suspend [i8 0, label %resume2
                                  i8 1, label %cleanup]

resume2:
  call void @print(i32 %inc1)
  call void @print(i32 %inc2)
  br label %cleanup

cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend
suspend:
  call i1 @llvm.coro.end(ptr %hdl, i1 0, token none)
  ret ptr %hdl
}

; Checks that we won't transform functions with optnone.
define ptr @f_optnone(i32 %n) presplitcoroutine optnone noinline {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)

  %inc1 = add i32 %n, 1
  %sp1 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp1, label %suspend [i8 0, label %resume1
                                  i8 1, label %cleanup]
resume1:
  %inc2 = add i32 %inc1, 1
  %sp2 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp2, label %suspend [i8 0, label %resume2
                                  i8 1, label %cleanup]

resume2:
  call void @print(i32 %inc1)
  call void @print(i32 %inc2)
  br label %cleanup

cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend
suspend:
  call i1 @llvm.coro.end(ptr %hdl, i1 0, token none)
  ret ptr %hdl
}

define ptr @f_multiple_remat(i32 %n) presplitcoroutine {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)

  %inc1 = add i32 %n, 1
  %inc2 = add i32 %inc1, 2
  %inc3 = add i32 %inc2, 3
  %inc4 = add i32 %inc3, 4
  %inc5 = add i32 %inc4, 5
  %inc6 = add i32 %inc5, 5
  %sp1 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp1, label %suspend [i8 0, label %resume1
                                  i8 1, label %cleanup]
resume1:
  %inc7 = add i32 %inc6, 1
  %sp2 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp2, label %suspend [i8 0, label %resume2
                                  i8 1, label %cleanup]

resume2:
  call void @print(i32 %inc1)
  call void @print(i32 %inc7)
  br label %cleanup

cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend
suspend:
  call i1 @llvm.coro.end(ptr %hdl, i1 0, token none)
  ret ptr %hdl
}

define ptr @f_common_def(i32 %n) presplitcoroutine {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)

  %inc1 = add i32 %n, 1
  %inc2 = add i32 %inc1, 2
  %inc3 = add i32 %n, 3
  %inc4 = add i32 %inc3, %inc1
  %inc5 = add i32 %inc4, %inc1
  %inc6 = add i32 %inc5, 5
  %sp1 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp1, label %suspend [i8 0, label %resume1
                                  i8 1, label %cleanup]
resume1:
  %inc7 = add i32 %inc6, 1
  %sp2 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp2, label %suspend [i8 0, label %resume2
                                  i8 1, label %cleanup]

resume2:
  call void @print(i32 %inc1)
  call void @print(i32 %inc7)
  br label %cleanup

cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend
suspend:
  call i1 @llvm.coro.end(ptr %hdl, i1 0, token none)
  ret ptr %hdl
}

define ptr @f_common_def_multi_result(i32 %n) presplitcoroutine {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)

  %inc1 = add i32 %n, 1
  %inc2 = add i32 %inc1, 2
  %inc3 = add i32 %n, 3
  %inc4 = add i32 %inc3, %inc1
  %inc5 = add i32 %inc4, %inc1
  %inc6 = add i32 %inc5, 4
  %inc7 = add i32 %inc6, 5
  %inc8 = add i32 %inc4, %inc2
  %inc9 = add i32 %inc8, 5
  %inc10 = add i32 %inc9, 6
  %inc11 = add i32 %inc10, 7
  %sp1 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp1, label %suspend [i8 0, label %resume1
                                  i8 1, label %cleanup]
resume1:
  %inc12 = add i32 %inc7, 1
  %sp2 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp2, label %suspend [i8 0, label %resume2
                                  i8 1, label %cleanup]

resume2:
  call void @print(i32 %inc11)
  call void @print(i32 %inc12)
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

declare noalias ptr @malloc(i32)
declare void @print(i32)
declare void @free(ptr)
