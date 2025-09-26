; In coro-split, this coroutine code reduced IR, produced using clang with async-exceptions
; crashed after first phase of fix because the terminator cleanupret was not implemented on predecessor fixer at the time
; RUN: opt < %s -passes='coro-split' -S

; Function Attrs: presplitcoroutine
define i8 @"?resuming_on_new_thread@@YA?AUtask@@V?$unique_ptr@HU?$default_delete@H@std@@@std@@0@Z"(ptr %0) #0 personality ptr null {
  invoke void @llvm.seh.scope.begin()
          to label %2 unwind label %15

2:                                                ; preds = %1
  %3 = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %4 = call ptr @llvm.coro.begin(token %3, ptr null)
  %5 = call token @llvm.coro.save(ptr null)
  %6 = call i8 @llvm.coro.suspend(token none, i1 false)
  invoke void @llvm.seh.try.begin()
          to label %7 unwind label %8

7:                                                ; preds = %2
  ret i8 0

8:                                                ; preds = %2
  %9 = catchswitch within none [label %10] unwind label %12

10:                                               ; preds = %8
  %11 = catchpad within %9 [ptr null, i32 0, ptr null]
  ret i8 0

12:                                               ; preds = %8
  %13 = cleanuppad within none []
  invoke void @llvm.seh.scope.end()
          to label %14 unwind label %15

14:                                               ; preds = %12
  ret i8 0

15:                                               ; preds = %12, %1
  %16 = cleanuppad within none []
  cleanupret from %16 unwind label %17

17:                                               ; preds = %15
  %18 = cleanuppad within none []
  store i32 0, ptr %0, align 4
  cleanupret from %18 unwind to caller
}

attributes #0 = { presplitcoroutine }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"eh-asynch", i32 1}
