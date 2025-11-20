; Tests that the readnone function which don't cross suspend points could be optimized expectly after split.
;
; RUN: opt < %s -S -passes='default<O3>' | FileCheck %s --check-prefixes=CHECK_SPLITTED
; RUN: opt < %s -S -passes='coro-split,early-cse,simplifycfg' | FileCheck %s --check-prefixes=CHECK_SPLITTED
; RUN: opt < %s -S -passes='coro-split,gvn,simplifycfg' | FileCheck %s --check-prefixes=CHECK_SPLITTED
; RUN: opt < %s -S -passes='coro-split,newgvn,simplifycfg' | FileCheck %s --check-prefixes=CHECK_SPLITTED
; RUN: opt < %s -S -passes='early-cse' | FileCheck %s --check-prefixes=CHECK_UNSPLITTED
; RUN: opt < %s -S -passes='gvn' | FileCheck %s --check-prefixes=CHECK_UNSPLITTED
; RUN: opt < %s -S -passes='newgvn' | FileCheck %s --check-prefixes=CHECK_UNSPLITTED

define ptr @f() presplitcoroutine {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  %sus_result = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sus_result, label %suspend [i8 0, label %resume
                                         i8 1, label %cleanup]
resume:
  %i = call i32 @readnone_func() readnone
  ; noop call to break optimization to combine two consecutive readonly calls.
  call void @nop()
  %j = call i32 @readnone_func() readnone
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
  call void @llvm.coro.end(ptr %hdl, i1 0, token none)
  ret ptr %hdl
}

;
; CHECK_SPLITTED-LABEL: f.resume(
; CHECK_SPLITTED-NEXT:  :
; CHECK_SPLITTED-NEXT:    call i32 @readnone_func() #[[ATTR_NUM:[0-9]+]]
; CHECK_SPLITTED-NEXT:    call void @nop()
; CHECK_SPLITTED-NEXT:    call void @print_same()
;
; CHECK_SPLITTED: attributes #[[ATTR_NUM]] = { memory(none) }
;
; CHECK_UNSPLITTED-LABEL: @f(
; CHECK_UNSPLITTED: br i1 %cmp, label %same, label %diff
; CHECK_UNSPLITTED-EMPTY:
; CHECK_UNSPLITTED-NEXT: same:
; CHECK_UNSPLITTED-NEXT:   call void @print_same()
; CHECK_UNSPLITTED-NEXT:   br label
; CHECK_UNSPLITTED-EMPTY:
; CHECK_UNSPLITTED-NEXT: diff:
; CHECK_UNSPLITTED-NEXT:   call void @print_diff()
; CHECK_UNSPLITTED-NEXT:   br label

declare i32 @readnone_func() readnone
declare void @nop()

declare void @print_same()
declare void @print_diff()
declare ptr @llvm.coro.free(token, ptr)
declare i32 @llvm.coro.size.i32()
declare i8  @llvm.coro.suspend(token, i1)

declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare i1 @llvm.coro.alloc(token)
declare ptr @llvm.coro.begin(token, ptr)
declare void @llvm.coro.end(ptr, i1, token)

declare noalias ptr @malloc(i32)
declare void @free(ptr)
