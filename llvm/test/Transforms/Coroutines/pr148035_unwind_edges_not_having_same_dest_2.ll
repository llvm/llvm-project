; In coro-split, this coroutine standard code reduced IR, produced using clang with async-exceptions
; crashed before fix because of the validation mismatch of Unwind edges out of a funclet pad must have the same unwind dest
; RUN: opt < %s -passes='coro-split' -S 2>&1 | FileCheck %s --implicit-check-not="Unwind edges out of a funclet pad must have the same unwind dest"

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.38.33135"

; Function Attrs: presplitcoroutine
define i1 @"?resuming_on_new_thread@@YA?AUtask@@AEAVjthread@std@@@Z"() #0 personality ptr null {
  %1 = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %2 = call ptr @llvm.coro.begin(token %1, ptr null)
  %3 = call token @llvm.coro.save(ptr null)
  %4 = call i8 @llvm.coro.suspend(token none, i1 false)
  invoke void @llvm.seh.try.begin()
          to label %common.ret unwind label %5

common.ret:                                       ; preds = %11, %13, %7, %0
  %common.ret.op = phi i1 [ false, %11], [false, %13 ], [false, %7], [false, %0]
  ret i1 %common.ret.op

5:                                                ; preds = %0
  %6 = catchswitch within none [label %7] unwind label %9

7:                                                ; preds = %5
  %8 = catchpad within %6 [ptr null, i32 0, ptr null]
  br label %common.ret

9:                                                ; preds = %5
  %10 = cleanuppad within none []
  invoke void @llvm.seh.scope.end() [ "funclet"(token %10) ]
          to label %11 unwind label %13

11:                                               ; preds = %9
  %12 = call i1 @llvm.coro.end(ptr null, i1 true, token none) [ "funclet"(token %10) ]
  br label %common.ret

13:                                               ; preds = %9
  %14 = cleanuppad within none []
  br label %common.ret
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare token @llvm.coro.id(i32, ptr readnone, ptr nocapture readonly, ptr) #1

; Function Attrs: nounwind
declare ptr @llvm.coro.begin(token, ptr writeonly) #2

; Function Attrs: nomerge nounwind
declare token @llvm.coro.save(ptr) #3

; Function Attrs: nounwind
declare i8 @llvm.coro.suspend(token, i1) #2

; Function Attrs: nounwind willreturn memory(write)
declare void @llvm.seh.try.begin() #4

; Function Attrs: nounwind memory(none)
declare void @llvm.seh.scope.end() #5

; Function Attrs: nounwind
declare i1 @llvm.coro.end(ptr, i1, token) #2

; uselistorder directives
uselistorder ptr null, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0 }

attributes #0 = { presplitcoroutine }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #2 = { nounwind }
attributes #3 = { nomerge nounwind }
attributes #4 = { nounwind willreturn memory(write) }
attributes #5 = { nounwind memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"eh-asynch", i32 1}
