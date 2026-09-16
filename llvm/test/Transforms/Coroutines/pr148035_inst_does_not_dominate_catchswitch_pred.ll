; Variant of pr148035_inst_does_not_dominate.ll where the last dominated
; block before the merged parameter cleanup is terminated by a catchswitch.
; A catchswitch must be the only non-PHI instruction in its block, so the
; frame reload cannot be materialized there; insertSpills hoists it to the
; nearest dominator that can hold it (here, the coro.begin block).
; RUN: opt < %s -passes='coro-split' -S | FileCheck %s

; The reload is hoisted above the catchswitch, into the block terminated by
; the seh.try.begin invoke, and the catchswitch's unwind edge feeds it into
; the merge PHI.
; CHECK-LABEL: define i8 @"?resuming_on_new_thread@@YA?AUtask@@Vunique_ptr@@@Z"(
; CHECK: %.reload = load ptr, ptr %.reload.addr
; CHECK-NEXT: invoke void @llvm.seh.try.begin()
; CHECK: catchswitch within none
; CHECK: %.pre.begin.merge = phi ptr [ %.reload, %{{.*}} ], [ %0, %{{.*}} ]
; CHECK-NEXT: %{{.*}} = cleanuppad within none []
; CHECK-NEXT: store i32 0, ptr %.pre.begin.merge

target triple = "x86_64-pc-windows-msvc"

; Function Attrs: presplitcoroutine
define i8 @"?resuming_on_new_thread@@YA?AUtask@@Vunique_ptr@@@Z"(ptr %0) #0 personality ptr null {
  invoke void @llvm.seh.scope.begin()
          to label %2 unwind label %11

2:                                                ; preds = %1
  %3 = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %4 = load volatile ptr, ptr null, align 8
  %5 = call ptr @llvm.coro.begin(token %3, ptr %4)
  %6 = call token @llvm.coro.save(ptr null)
  %7 = call i8 @llvm.coro.suspend(token none, i1 false)
  invoke void @llvm.seh.try.begin()
          to label %common.ret unwind label %8

common.ret:                                       ; preds = %10, %2
  ret i8 0

8:                                                ; preds = %2
  %9 = catchswitch within none [label %10] unwind label %11

10:                                               ; preds = %8
  %catch.pad = catchpad within %9 [ptr null, i32 0, ptr null]
  catchret from %catch.pad to label %common.ret

11:                                               ; preds = %8, %1
  %12 = cleanuppad within none []
  store i32 0, ptr %0, align 4
  cleanupret from %12 unwind to caller
}

attributes #0 = { presplitcoroutine }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"eh-asynch", i32 1}
