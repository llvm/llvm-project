; Coroutine calls marked with `coro_elide_safe` should be elided.
; Inside `caller`, we expect the `callee` coroutine to be elided.
; Inside `caller_conditional`, `callee` is only called on an unlikely
; path, hence we expect the `callee` coroutine NOT to be elided.
;
; RUN: opt < %s -S -passes='cgscc(coro-annotation-elide)' | FileCheck %s

%struct.Task = type { ptr }

declare void @print(i32) nounwind

; resume part of the coroutine
define fastcc void @callee.resume(ptr dereferenceable(1)) {
  tail call void @print(i32 0)
  ret void
}

; destroy part of the coroutine
define fastcc void @callee.destroy(ptr) {
  tail call void @print(i32 1)
  ret void
}

; cleanup part of the coroutine
define fastcc void @callee.cleanup(ptr) {
  tail call void @print(i32 2)
  ret void
}

@callee.resumers = internal constant [3 x ptr] [
  ptr @callee.resume, ptr @callee.destroy, ptr @callee.cleanup]

declare void @alloc(i1) nounwind

; CHECK-LABEL: define ptr @callee
define ptr @callee(i8 %arg) {
entry:
  %task = alloca %struct.Task, align 8
  %id = call token @llvm.coro.id(i32 0, ptr null,
                          ptr @callee,
                          ptr @callee.resumers)
  %alloc = call i1 @llvm.coro.alloc(token %id)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr null)
  store ptr %hdl, ptr %task
  ret ptr %task
}

; CHECK-LABEL: define ptr @callee.noalloc
define ptr @callee.noalloc(i8 %arg, ptr dereferenceable(32) align(8) %frame) {
 entry:
  %task = alloca %struct.Task, align 8
  %id = call token @llvm.coro.id(i32 0, ptr null,
                          ptr @callee,
                          ptr @callee.resumers)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr null)
  store ptr %hdl, ptr %task
  ret ptr %task
}

; CHECK-LABEL: define ptr @caller()
; Function Attrs: presplitcoroutine
define ptr @caller() #0 {
entry:
  %task = call ptr @callee(i8 0) coro_elide_safe
  ret ptr %task
  ; CHECK: %[[TASK:.+]] = alloca %struct.Task, align 8
  ; CHECK-NEXT: %[[FRAME:.+]] = alloca [32 x i8], align 8
  ; CHECK-NEXT: call void @llvm.lifetime.start.p0(ptr %[[TASK]])
  ; CHECK-NEXT: %[[ID:.+]] = call token @llvm.coro.id(i32 0, ptr null, ptr @callee, ptr @callee.resumers)
  ; CHECK-NEXT: %[[HDL:.+]] = call ptr @llvm.coro.begin(token %[[ID]], ptr null)
  ; CHECK-NEXT: store ptr %[[HDL]], ptr %[[TASK]], align 8
  ; CHECK-NEXT: call void @llvm.lifetime.end.p0(ptr %[[TASK]])
  ; CHECK-NEXT: ret ptr %[[TASK]]
}

; CHECK-LABEL: define ptr @caller_conditional(i1 %cond)
; Function Attrs: presplitcoroutine
define ptr @caller_conditional(i1 %cond) #0 {
entry:
  br i1 %cond, label %call, label %ret

call:
  ; CHECK-NOT: alloca
  ; CHECK-NOT: @llvm.coro.id({{.*}}, ptr @callee, {{.*}})
  ; CHECK: %task = call ptr @callee(i8 0)
  ; CHECK-NEXT: br label %ret
  %task = call ptr @callee(i8 0) coro_elide_safe
  br label %ret

ret:
  %retval = phi ptr [ %task, %call ], [ null, %entry ]
  ret ptr %retval
}

declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare ptr @llvm.coro.begin(token, ptr)
declare ptr @llvm.coro.frame()
declare ptr @llvm.coro.subfn.addr(ptr, i8)
declare i1 @llvm.coro.alloc(token)

attributes #0 = { presplitcoroutine }
