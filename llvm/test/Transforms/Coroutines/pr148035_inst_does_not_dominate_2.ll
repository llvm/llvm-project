; In coro-split, this coroutine code reduced IR, produced using clang with async-exceptions
; crashed before fix because of the validation mismatch of Instruction does not dominate all uses!
; RUN: opt < %s -passes='coro-split' -S

; Function Attrs: presplitcoroutine
define i8 @"?resuming_on_new_thread@@YA?AUtask@@Vunique_ptr@@@Z"(ptr %0) #0 personality ptr null {
  invoke void @llvm.seh.scope.begin()
          to label %2 unwind label %14

2:                                                ; preds = %1
  %3 = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %4 = load volatile ptr, ptr null, align 8
  %5 = call ptr @llvm.coro.begin(token %3, ptr %4)
  %6 = call token @llvm.coro.save(ptr null)
  %7 = call i8 @llvm.coro.suspend(token none, i1 false)
  invoke void @llvm.seh.try.begin()
          to label %common.ret unwind label %8

common.ret:                                       ; preds = %12, %10, %2
  ret i8 0

8:                                                ; preds = %2
  %9 = catchswitch within none [label %10] unwind label %12

10:                                               ; preds = %8
  %11 = catchpad within %9 [ptr null, i32 0, ptr null]
  br label %common.ret

12:                                               ; preds = %8
  %13 = cleanuppad within none []
  invoke void @llvm.seh.scope.end()
          to label %common.ret unwind label %14

14:                                               ; preds = %12, %1
  %15 = cleanuppad within none []
  store i32 0, ptr %0, align 4
  br label %common.ret
}

attributes #0 = { presplitcoroutine }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"eh-asynch", i32 1}
