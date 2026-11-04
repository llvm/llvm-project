; Hand-trimmed from real clang /EHa output for pr148035 (unlike the other
; pr148035 tests, this keeps the structure llvm-reduce stripped): the
; coro.alloc/operator-new invoke, the unwind coro.end cleanup chain, and the
; coro.free/operator-delete path. The parameter-destructor cleanup
; (%param.cleanup) is reachable from two pre-coro.begin edges (the entry
; seh.scope.begin fault edge and operator new's unwind edge) and from two
; post-coro.begin unwind chains, so its merge PHI has two raw-argument
; entries and two frame-reload entries.
; RUN: opt < %s -passes='coro-split' -S | FileCheck %s

; CHECK-LABEL: define i8 @"?resuming_on_new_thread@@YA?AUtask@@Uunique_ptr@@@Z"(
; CHECK: %.pre.begin.merge = phi ptr [ %.reload{{.*}}, %{{.*}} ], [ %.reload{{.*}}, %{{.*}} ], [ %0, %coro.alloc ], [ %0, %entry ]
; CHECK-NEXT: %{{.*}} = cleanuppad within none []
; CHECK-NEXT: store i32 0, ptr %.pre.begin.merge

target triple = "x86_64-pc-windows-msvc"

; Function Attrs: presplitcoroutine
define i8 @"?resuming_on_new_thread@@YA?AUtask@@Uunique_ptr@@@Z"(ptr %0) #0 personality ptr @__CxxFrameHandler3 {
entry:
  invoke void @llvm.seh.scope.begin()
          to label %coro.check.alloc unwind label %param.cleanup

coro.check.alloc:                                 ; preds = %entry
  %id = call token @llvm.coro.id(i32 16, ptr null, ptr @"?resuming_on_new_thread@@YA?AUtask@@Uunique_ptr@@@Z", ptr null)
  %need.alloc = call i1 @llvm.coro.alloc(token %id)
  br i1 %need.alloc, label %coro.alloc, label %coro.init

coro.alloc:                                       ; preds = %coro.check.alloc
  %size = call i64 @llvm.coro.size.i64()
  %mem = invoke ptr @"??2@YAPEAX_K@Z"(i64 %size)
          to label %coro.init unwind label %param.cleanup

coro.init:                                        ; preds = %coro.alloc, %coro.check.alloc
  %phi.mem = phi ptr [ null, %coro.check.alloc ], [ %mem, %coro.alloc ]
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %phi.mem)
  %save = call token @llvm.coro.save(ptr null)
  %suspend = call i8 @llvm.coro.suspend(token %save, i1 false)
  switch i8 %suspend, label %coro.ret [
    i8 0, label %resume
    i8 1, label %cleanup
  ]

resume:                                           ; preds = %coro.init
  invoke void @"?body@@YAXXZ"()
          to label %cleanup unwind label %eh.body

cleanup:                                          ; preds = %resume, %coro.init
  %free.mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @"??3@YAXPEAX@Z"(ptr %free.mem)
  br label %coro.ret

coro.ret:                                         ; preds = %cleanup, %coro.init
  call void @llvm.coro.end(ptr null, i1 false, token none)
  invoke void @llvm.seh.scope.end()
          to label %ret unwind label %param.cleanup

ret:                                              ; preds = %coro.ret
  store i32 0, ptr %0, align 4
  ret i8 0

eh.body:                                          ; preds = %resume
  %pad.body = cleanuppad within none []
  invoke void @llvm.seh.scope.end() [ "funclet"(token %pad.body) ]
          to label %coro.end.eh unwind label %eh.free

coro.end.eh:                                      ; preds = %eh.body
  call void @llvm.coro.end(ptr null, i1 true, token none) [ "funclet"(token %pad.body) ]
  cleanupret from %pad.body unwind label %eh.free

eh.free:                                          ; preds = %coro.end.eh, %eh.body
  %pad.free = cleanuppad within none []
  %free.mem.eh = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @"??3@YAXPEAX@Z"(ptr %free.mem.eh) [ "funclet"(token %pad.free) ]
  cleanupret from %pad.free unwind label %param.cleanup

param.cleanup:                                    ; preds = %eh.free, %coro.ret, %coro.alloc, %entry
  %pad.param = cleanuppad within none []
  store i32 0, ptr %0, align 4
  cleanupret from %pad.param unwind to caller
}

declare void @"?body@@YAXXZ"()
declare ptr @"??2@YAPEAX_K@Z"(i64)
declare void @"??3@YAXPEAX@Z"(ptr)
declare i32 @__CxxFrameHandler3(...)

attributes #0 = { presplitcoroutine }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"eh-asynch", i32 1}
