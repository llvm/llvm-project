; In coro-split, this coroutine standard code reduced IR, produced using clang with async-exceptions
; crashed before fix because of the validation mismatch of Unwind edges out of a funclet pad must have the same unwind dest
; RUN: opt < %s -passes='coro-split' -S 2>&1 | FileCheck %s --implicit-check-not="Unwind edges out of a funclet pad must have the same unwind dest"

; Function Attrs: presplitcoroutine
define i8 @"?resuming_on_new_thread@@YA?AUtask@@AEAVjthread@std@@@Z"() #0 personality ptr null {
  %1 = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %2 = call ptr @llvm.coro.begin(token %1, ptr null)
  %3 = call token @llvm.coro.save(ptr null)
  %4 = call i8 @llvm.coro.suspend(token none, i1 false)
  invoke void @llvm.seh.try.begin()
          to label %common.ret unwind label %5

common.ret:                                       ; preds = %13, %7, %0
  ret i8 0

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
  cleanupret from %10 unwind label %13

13:                                               ; preds = %11, %9
  %14 = cleanuppad within none []
  br label %common.ret
}

attributes #0 = { presplitcoroutine }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"eh-asynch", i32 1}
