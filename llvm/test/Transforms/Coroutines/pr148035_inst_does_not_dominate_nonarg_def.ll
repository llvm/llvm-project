; Variant of pr148035_inst_does_not_dominate.ll where the spilled value is
; not an argument but an instruction computed before coro.begin (%p). The
; SSA repair registers the original value at its defining block rather than
; the entry block; the merge PHI receives %p on the pre-coro.begin edge and
; the frame reload on the post-coro.begin edge.
; RUN: opt < %s -passes='coro-split' -S | FileCheck %s

; CHECK-LABEL: define i8 @"?resuming_on_new_thread@@YA?AUtask@@Vunique_ptr@@@Z"(
; CHECK: %p.reload = load ptr, ptr %p.reload.addr
; CHECK-NEXT: invoke void @llvm.seh.scope.end()
; CHECK: %p.pre.begin.merge = phi ptr [ %p.reload, %{{.*}} ], [ %p, %{{.*}} ]
; CHECK-NEXT: %{{.*}} = cleanuppad within none []
; CHECK-NEXT: store i32 0, ptr %p.pre.begin.merge

target triple = "x86_64-pc-windows-msvc"

; Function Attrs: presplitcoroutine
define i8 @"?resuming_on_new_thread@@YA?AUtask@@Vunique_ptr@@@Z"(ptr %0) #0 personality ptr null {
  %p = load ptr, ptr %0, align 8
  invoke void @llvm.seh.scope.begin()
          to label %2 unwind label %14

2:                                                ; preds = %1
  %3 = call token @llvm.coro.id(i32 0, ptr null, ptr @"?resuming_on_new_thread@@YA?AUtask@@Vunique_ptr@@@Z", ptr null)
  %4 = load volatile ptr, ptr null, align 8
  %5 = call ptr @llvm.coro.begin(token %3, ptr %4)
  %6 = call token @llvm.coro.save(ptr null)
  %7 = call i8 @llvm.coro.suspend(token none, i1 false)
  invoke void @llvm.seh.try.begin()
          to label %common.ret unwind label %8

common.ret:                                       ; preds = %10, %2
  ret i8 0

cleanup.ret:                                      ; preds = %12
  cleanupret from %13 unwind to caller

8:                                                ; preds = %2
  %9 = catchswitch within none [label %10] unwind label %12

10:                                               ; preds = %8
  %11 = catchpad within %9 [ptr null, i32 0, ptr null]
  catchret from %11 to label %common.ret

12:                                               ; preds = %8
  %13 = cleanuppad within none []
  invoke void @llvm.seh.scope.end()
          to label %cleanup.ret unwind label %14

14:                                               ; preds = %12, %1
  %15 = cleanuppad within none []
  store i32 0, ptr %p, align 4
  cleanupret from %15 unwind to caller
}

attributes #0 = { presplitcoroutine }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"eh-asynch", i32 1}
