; Tests that the readnone function which cross suspend points wouldn't be misoptimized.
; RUN: opt < %s -S -passes='default<O3>' | FileCheck %s --check-prefixes=CHECK,CHECK_SPLITTED
; RUN: opt < %s -S -passes='early-cse' | FileCheck %s --check-prefixes=CHECK,CHECK_UNSPLITTED
; RUN: opt < %s -S -passes='gvn' | FileCheck %s --check-prefixes=CHECK,CHECK_UNSPLITTED
; RUN: opt < %s -S -passes='newgvn' | FileCheck %s --check-prefixes=CHECK,CHECK_UNSPLITTED

define ptr @f() presplitcoroutine {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  %j = call i32 @readnone_func() readnone
  %sus_result = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sus_result, label %suspend [i8 0, label %resume
                                         i8 1, label %cleanup]
resume:
  %i = call i32 @readnone_func() readnone
  %cmp = icmp eq i32 %i, %j
  br i1 %cmp, label %same, label %diff

same:
  call void @print_same()
  br label %cleanup

diff:
  call void @print_diff()
  br label %cleanup

cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend

suspend:
  call i1 @llvm.coro.end(ptr %hdl, i1 0)
  ret ptr %hdl
}

; Tests that normal functions wouldn't be affected.
define i1 @normal_function() {
entry:
  %i = call i32 @readnone_func() readnone
  %j = call i32 @readnone_func() readnone
  %cmp = icmp eq i32 %i, %j
  br i1 %cmp, label %same, label %diff

same:
  call void @print_same()
  ret i1 true

diff:
  call void @print_diff()
  ret i1 false
}

; CHECK_SPLITTED-LABEL: normal_function(
; CHECK_SPLITTED-NEXT: entry
; CHECK_SPLITTED-NEXT:   call i32 @readnone_func()
; CHECK_SPLITTED-NEXT:   call void @print_same()
; CHECK_SPLITTED-NEXT:   ret i1 true
;
; CHECK_SPLITTED-LABEL: f.resume(
; CHECK_UNSPLITTED-LABEL: @f(
; CHECK: br i1 %cmp, label %same, label %diff
; CHECK-EMPTY:
; CHECK-NEXT: same:
; CHECK-NEXT:   call void @print_same()
; CHECK-NEXT:   br label
; CHECK-EMPTY:
; CHECK-NEXT: diff:
; CHECK-NEXT:   call void @print_diff()
; CHECK-NEXT:   br label

declare i32 @readnone_func() readnone

declare void @print_same()
declare void @print_diff()
declare ptr @llvm.coro.free(token, ptr)
declare i32 @llvm.coro.size.i32()
declare i8  @llvm.coro.suspend(token, i1)

declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare i1 @llvm.coro.alloc(token)
declare ptr @llvm.coro.begin(token, ptr)
declare i1 @llvm.coro.end(ptr, i1)

declare noalias ptr @malloc(i32)
declare void @free(ptr)
