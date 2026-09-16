; Variant of pr148035_inst_does_not_dominate.ll with two chained merged
; cleanups, each reachable from its own pre-coro.begin edge (the entry
; seh.scope.begin fault edge into %padA, the allocation invoke's unwind edge
; into %padB) as well as from post-coro.begin paths. SSA repair must insert
; cascading PHIs: the merge in %padB receives the merge of %padA on the
; chained cleanupret edge.
; RUN: opt < %s -passes='coro-split' -S | FileCheck %s

; CHECK-LABEL: define i8 @"?resuming_on_new_thread@@YA?AUtask@@Vunique_ptr@@@Z"(
; CHECK: %[[MERGEA:.*pre.begin.merge.*]] = phi ptr [ %.reload, %{{.*}} ], [ %0, %{{.*}} ]
; CHECK-NEXT: %{{.*}} = cleanuppad within none []
; CHECK-NEXT: store i32 1, ptr %[[MERGEA]]
; CHECK: %[[MERGEB:.*pre.begin.merge.*]] = phi ptr [ %[[MERGEA]], %padA ], [ %0, %coro.alloc ]
; CHECK-NEXT: %{{.*}} = cleanuppad within none []
; CHECK-NEXT: store i32 0, ptr %[[MERGEB]]

target triple = "x86_64-pc-windows-msvc"

; Function Attrs: presplitcoroutine
define i8 @"?resuming_on_new_thread@@YA?AUtask@@Vunique_ptr@@@Z"(ptr %0) #0 personality ptr null {
entry:
  invoke void @llvm.seh.scope.begin()
          to label %coro.check.alloc unwind label %padA

coro.check.alloc:                                 ; preds = %entry
  %id = call token @llvm.coro.id(i32 16, ptr null, ptr null, ptr null)
  %need.alloc = call i1 @llvm.coro.alloc(token %id)
  br i1 %need.alloc, label %coro.alloc, label %coro.init

coro.alloc:                                       ; preds = %coro.check.alloc
  %size = call i64 @llvm.coro.size.i64()
  %mem = invoke ptr @"??2@YAPEAX_K@Z"(i64 %size)
          to label %coro.init unwind label %padB

coro.init:                                        ; preds = %coro.alloc, %coro.check.alloc
  %phi.mem = phi ptr [ null, %coro.check.alloc ], [ %mem, %coro.alloc ]
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %phi.mem)
  %save = call token @llvm.coro.save(ptr null)
  %suspend = call i8 @llvm.coro.suspend(token %save, i1 false)
  invoke void @llvm.seh.try.begin()
          to label %common.ret unwind label %postpad

common.ret:                                       ; preds = %coro.init
  ret i8 0

postpad:                                          ; preds = %coro.init
  %pp = cleanuppad within none []
  cleanupret from %pp unwind label %padA

padA:                                             ; preds = %postpad, %entry
  %pa = cleanuppad within none []
  store i32 1, ptr %0, align 4
  cleanupret from %pa unwind label %padB

padB:                                             ; preds = %padA, %coro.alloc
  %pb = cleanuppad within none []
  store i32 0, ptr %0, align 4
  cleanupret from %pb unwind to caller
}

declare ptr @"??2@YAPEAX_K@Z"(i64)

attributes #0 = { presplitcoroutine }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"eh-asynch", i32 1}
