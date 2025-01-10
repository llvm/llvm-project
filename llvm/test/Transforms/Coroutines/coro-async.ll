; RUN: opt < %s -passes='default<O2>' -S | FileCheck --check-prefixes=CHECK %s
; RUN: opt < %s -O0 -S | FileCheck --check-prefixes=CHECK-O0 %s
target datalayout = "p:64:64:64"

%async.task = type { i64 }
%async.actor = type { i64 }
%async.fp = type <{ i32, i32 }>

%async.ctxt = type { ptr, ptr }

; The async callee.
@my_other_async_function_fp = external global <{ i32, i32 }>
declare void @my_other_async_function(ptr %async.ctxt)

; The current async function (the caller).
; This struct describes an async function. The first field is the
; relative offset to the async function implementation, the second field is the
; size needed for the async context of the current async function.

@my_async_function_fp = constant <{ i32, i32 }>
  <{ i32 trunc ( ; Relative pointer to async function
       i64 sub (
         i64 ptrtoint (ptr @my_async_function to i64),
         i64 ptrtoint (ptr getelementptr inbounds (<{ i32, i32 }>, ptr @my_async_function_fp, i32 0, i32 1) to i64)
       )
     to i32),
     i32 128    ; Initial async context size without space for frame
}>
@my_async_function_pa_fp = constant <{ i32, i32 }>
  <{ i32 trunc (
       i64 sub (
         i64 ptrtoint (ptr @my_async_function_pa to i64),
         i64 ptrtoint (ptr getelementptr inbounds (<{ i32, i32 }>, ptr @my_async_function_pa_fp, i32 0, i32 1) to i64)
       )
     to i32),
     i32 8
}>

; Function that implements the dispatch to the callee function.
define swiftcc void @my_async_function.my_other_async_function_fp.apply(ptr %fnPtr, ptr %async.ctxt, ptr %task, ptr %actor) {
  tail call swiftcc void %fnPtr(ptr %async.ctxt, ptr %task, ptr %actor)
  ret void
}

declare void @some_user(i64)
declare void @some_may_write(ptr)

define ptr @__swift_async_resume_project_context(ptr %ctxt) {
entry:
  %resume_ctxt = load ptr, ptr %ctxt, align 8
  ret ptr %resume_ctxt
}

define ptr @resume_context_projection(ptr %ctxt) {
entry:
  %resume_ctxt = load ptr, ptr %ctxt, align 8
  ret ptr %resume_ctxt
}


define swiftcc void @my_async_function(ptr swiftasync %async.ctxt, ptr %task, ptr %actor) presplitcoroutine !dbg !1 {
entry:
  %tmp = alloca { i64, i64 }, align 8
  %vector = alloca <4 x double>, align 16
  %proj.1 = getelementptr inbounds { i64, i64 }, ptr %tmp, i64 0, i32 0
  %proj.2 = getelementptr inbounds { i64, i64 }, ptr %tmp, i64 0, i32 1

  %id = call token @llvm.coro.id.async(i32 128, i32 16, i32 0,
          ptr @my_async_function_fp)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr null)
  store i64 0, ptr %proj.1, align 8
  store i64 1, ptr %proj.2, align 8
  call void @some_may_write(ptr %proj.1)

	; Begin lowering: apply %my_other_async_function(%args...)

  ; setup callee context
  %callee_context = call ptr @llvm.coro.async.context.alloc(ptr %task, ptr @my_other_async_function_fp)
  ; store arguments ...
  ; ... (omitted)

  ; store the return continuation
  %callee_context.return_to_caller.addr = getelementptr inbounds %async.ctxt, ptr %callee_context, i32 0, i32 1
  %resume.func_ptr = call ptr @llvm.coro.async.resume()
  store ptr %resume.func_ptr, ptr %callee_context.return_to_caller.addr

  ; store caller context into callee context
  store ptr %async.ctxt, ptr %callee_context
  %vector_spill = load <4 x double>, ptr %vector, align 16
  %res = call {ptr, ptr, ptr} (i32, ptr, ptr, ...) @llvm.coro.suspend.async(i32 0,
                                                  ptr %resume.func_ptr,
                                                  ptr @__swift_async_resume_project_context,
                                                  ptr @my_async_function.my_other_async_function_fp.apply,
                                                  ptr @asyncSuspend, ptr %callee_context, ptr %task, ptr %actor), !dbg !5

  call void @llvm.coro.async.context.dealloc(ptr %callee_context)
  %continuation_task_arg = extractvalue {ptr, ptr, ptr} %res, 1
  %val = load i64, ptr %proj.1
  call void @some_user(i64 %val)
  %val.2 = load i64, ptr %proj.2
  call void @some_user(i64 %val.2)
  store <4 x double> %vector_spill, ptr %vector, align 16
  tail call swiftcc void @asyncReturn(ptr %async.ctxt, ptr %continuation_task_arg, ptr %actor)
  call i1 (ptr, i1, ...) @llvm.coro.end.async(ptr %hdl, i1 0)
  unreachable
}

define void @my_async_function_pa(ptr %ctxt, ptr %task, ptr %actor) {
  call void @llvm.coro.async.size.replace(ptr @my_async_function_pa_fp, ptr @my_async_function_fp)
  call swiftcc void @my_async_function(ptr %ctxt, ptr %task, ptr %actor)
  ret void
}

; Make sure we update the async function pointer
; CHECK: @my_async_function_fp = constant <{ i32, i32 }> <{ {{.*}}, i32 176 }
; CHECK: @my_async_function_pa_fp = constant <{ i32, i32 }> <{ {{.*}}, i32 176 }
; CHECK: @my_async_function2_fp = constant <{ i32, i32 }> <{ {{.*}}, i32 176 }

; CHECK-LABEL: define swiftcc void @my_async_function(ptr swiftasync initializes((152, 160)) %async.ctxt, ptr %task, ptr %actor)
; CHECK-O0-LABEL: define swiftcc void @my_async_function(ptr swiftasync %async.ctxt, ptr %task, ptr %actor)
; CHECK-SAME: !dbg ![[SP1:[0-9]+]] {
; CHECK: coro.return:
; CHECK:   [[FRAMEPTR:%.*]] = getelementptr inbounds nuw i8, ptr %async.ctxt, i64 128
; CHECK:   [[ACTOR_SPILL_ADDR:%.*]] = getelementptr inbounds nuw i8, ptr %async.ctxt, i64 152
; CHECK:   store ptr %actor, ptr [[ACTOR_SPILL_ADDR]]
; CHECK:   [[ADDR1:%.*]]  = getelementptr inbounds nuw i8, ptr %async.ctxt, i64 144
; CHECK:   store ptr %async.ctxt, ptr [[ADDR1]]
; CHECK:   [[ALLOCA_PRJ2:%.*]] = getelementptr inbounds nuw i8, ptr %async.ctxt, i64 136
; CHECK:   store i64 0, ptr [[FRAMEPTR]]
; CHECK:   store i64 1, ptr [[ALLOCA_PRJ2]]
; CHECK:   tail call void @some_may_write(ptr nonnull [[FRAMEPTR]])
; CHECK:   [[CALLEE_CTXT:%.*]] = tail call ptr @llvm.coro.async.context.alloc(ptr %task, ptr nonnull @my_other_async_function_fp)
; CHECK:   [[CALLEE_CTXT_SPILL:%.*]] = getelementptr inbounds nuw i8, ptr %async.ctxt, i64 160
; CHECK:   store ptr [[CALLEE_CTXT]], ptr [[CALLEE_CTXT_SPILL]]
; CHECK:   [[TYPED_RETURN_TO_CALLER_ADDR:%.*]] = getelementptr inbounds nuw i8, ptr [[CALLEE_CTXT]], i64 8
; CHECK:   store ptr @my_async_functionTQ0_, ptr [[TYPED_RETURN_TO_CALLER_ADDR]]
; CHECK:   store ptr %async.ctxt, ptr [[CALLEE_CTXT]]
; Make sure the spill is underaligned to the max context alignment (16).
; CHECK-O0:   [[VECTOR_SPILL:%.*]] = load <4 x double>, ptr {{.*}}
; CHECK-O0:   [[VECTOR_SPILL_ADDR:%.*]] = getelementptr inbounds %my_async_function.Frame, ptr {{.*}}, i32 0, i32 1
; CHECK-O0:   store <4 x double> [[VECTOR_SPILL]], ptr [[VECTOR_SPILL_ADDR]], align 16
; CHECK:   tail call swiftcc void @asyncSuspend(ptr nonnull [[CALLEE_CTXT]], ptr %task, ptr %actor)
; CHECK:   ret void
; CHECK: }

; CHECK-LABEL: define internal swiftcc void @my_async_functionTQ0_(ptr nocapture readonly swiftasync %0, ptr %1, ptr nocapture readnone %2)
; CHECK-O0-LABEL: define internal swiftcc void @my_async_functionTQ0_(ptr swiftasync %0, ptr %1, ptr %2)
; CHECK-SAME: !dbg ![[SP2:[0-9]+]] {
; CHECK: entryresume.0:
; CHECK:   [[CALLER_CONTEXT:%.*]] = load ptr, ptr %0
; CHECK:   [[FRAME_PTR:%.*]] = getelementptr inbounds nuw i8, ptr [[CALLER_CONTEXT]], i64 128
; CHECK-O0:   [[VECTOR_SPILL_ADDR:%.*]] = getelementptr inbounds %my_async_function.Frame, ptr {{.*}}, i32 0, i32 1
; CHECK-O0:   load <4 x double>, ptr [[VECTOR_SPILL_ADDR]], align 16
; CHECK:   [[CALLEE_CTXT_SPILL_ADDR:%.*]] = getelementptr inbounds nuw i8, ptr [[CALLER_CONTEXT]], i64 160
; CHECK:   [[CALLEE_CTXT_RELOAD:%.*]] = load ptr, ptr [[CALLEE_CTXT_SPILL_ADDR]]
; CHECK:   [[ACTOR_RELOAD_ADDR:%.*]] = getelementptr inbounds nuw i8, ptr [[CALLER_CONTEXT]], i64 152
; CHECK:   [[ACTOR_RELOAD:%.*]] = load ptr, ptr [[ACTOR_RELOAD_ADDR]]
; CHECK:   [[ADDR1:%.*]] = getelementptr inbounds nuw i8, ptr [[CALLER_CONTEXT]], i64 144
; CHECK:   [[ASYNC_CTXT_RELOAD:%.*]] = load ptr, ptr [[ADDR1]]
; CHECK:   [[ALLOCA_PRJ2:%.*]] = getelementptr inbounds nuw i8, ptr [[CALLER_CONTEXT]], i64 136
; CHECK:   tail call void @llvm.coro.async.context.dealloc(ptr nonnull [[CALLEE_CTXT_RELOAD]])
; CHECK:   [[VAL1:%.*]] = load i64, ptr [[FRAME_PTR]]
; CHECK:   tail call void @some_user(i64 [[VAL1]])
; CHECK:   [[VAL2:%.*]] = load i64, ptr [[ALLOCA_PRJ2]]
; CHECK:   tail call void @some_user(i64 [[VAL2]])
; CHECK:   tail call swiftcc void @asyncReturn(ptr [[ASYNC_CTXT_RELOAD]], ptr %1, ptr [[ACTOR_RELOAD]])
; CHECK:   ret void
; CHECK: }

@my_async_function2_fp = constant <{ i32, i32 }>
  <{ i32 trunc ( ; Relative pointer to async function
       i64 sub (
         i64 ptrtoint (ptr @my_async_function2 to i64),
         i64 ptrtoint (ptr getelementptr inbounds (<{ i32, i32 }>, ptr @my_async_function2_fp, i32 0, i32 1) to i64)
       )
     to i32),
     i32 128    ; Initial async context size without space for frame
  }>

define swiftcc void @my_async_function2(ptr %task, ptr %actor, ptr %async.ctxt) presplitcoroutine "frame-pointer"="all" !dbg !6 {
entry:

  %id = call token @llvm.coro.id.async(i32 128, i32 16, i32 2, ptr @my_async_function2_fp)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr null)
  ; setup callee context
  %callee_context = call ptr @llvm.coro.async.context.alloc(ptr %task, ptr @my_other_async_function_fp)

  %callee_context.return_to_caller.addr = getelementptr inbounds %async.ctxt, ptr %callee_context, i32 0, i32 1
  %resume.func_ptr = call ptr @llvm.coro.async.resume()
  store ptr %resume.func_ptr, ptr %callee_context.return_to_caller.addr
  store ptr %async.ctxt, ptr %callee_context
  %res = call {ptr, ptr, ptr} (i32, ptr, ptr, ...) @llvm.coro.suspend.async(i32 2,
                                                  ptr %resume.func_ptr,
                                                  ptr @resume_context_projection,
                                                  ptr @my_async_function.my_other_async_function_fp.apply,
                                                  ptr @asyncSuspend, ptr %callee_context, ptr %task, ptr %actor), !dbg !9

  %continuation_task_arg = extractvalue {ptr, ptr, ptr} %res, 0

  %callee_context.return_to_caller.addr.1 = getelementptr inbounds %async.ctxt, ptr %callee_context, i32 0, i32 1
  %resume.func_ptr.1 = call ptr @llvm.coro.async.resume()
  store ptr %resume.func_ptr.1, ptr %callee_context.return_to_caller.addr.1
  store ptr %async.ctxt, ptr %callee_context
  %res.2 = call {ptr, ptr, ptr} (i32, ptr, ptr, ...) @llvm.coro.suspend.async(i32 0,
                                                  ptr %resume.func_ptr.1,
                                                  ptr @resume_context_projection,
                                                  ptr @my_async_function.my_other_async_function_fp.apply,
                                                  ptr @asyncSuspend, ptr %callee_context, ptr %task, ptr %actor)

  call void @llvm.coro.async.context.dealloc(ptr %callee_context)
  %continuation_actor_arg = extractvalue {ptr, ptr, ptr} %res.2, 1

  tail call swiftcc void @asyncReturn(ptr %async.ctxt, ptr %continuation_task_arg, ptr %continuation_actor_arg)
  call i1 @llvm.coro.end(ptr %hdl, i1 0, token none)
  unreachable
}

; CHECK-LABEL: define swiftcc void @my_async_function2(ptr %task, ptr %actor, ptr %async.ctxt)
; CHECK-SAME: #[[FRAMEPOINTER:[0-9]+]]
; CHECK-SAME: !dbg ![[SP3:[0-9]+]]
; CHECK: store ptr %async.ctxt,
; CHECK: store ptr %actor,
; CHECK: store ptr %task,
; CHECK: [[CALLEE_CTXT:%.*]] =  tail call ptr @llvm.coro.async.context.alloc(
; CHECK: store ptr [[CALLEE_CTXT]],
; CHECK: store ptr @my_async_function2.resume.0,
; CHECK: store ptr %async.ctxt,
; CHECK: tail call swiftcc void @asyncSuspend(ptr nonnull [[CALLEE_CTXT]], ptr %task, ptr %actor)
; CHECK: ret void

; CHECK-LABEL: define internal swiftcc void @my_async_function2.resume.0(ptr %0, ptr nocapture readnone %1, ptr nocapture readonly %2)
; CHECK-SAME: #[[FRAMEPOINTER]]
; CHECK-SAME: !dbg ![[SP4:[0-9]+]]
; CHECK: [[CALLEE_CTXT:%.*]] = load ptr, ptr %2
; CHECK: [[CALLEE_CTXT_SPILL_ADDR:%.*]] = getelementptr inbounds nuw i8, ptr [[CALLEE_CTXT]], i64 152
; CHECK: store ptr @my_async_function2.resume.1,
; CHECK: [[CALLLE_CTXT_RELOAD:%.*]] = load ptr, ptr [[CALLEE_CTXT_SPILL_ADDR]]
; CHECK: tail call swiftcc void @asyncSuspend(ptr [[CALLEE_CTXT_RELOAD]]
; CHECK: ret void

; CHECK-LABEL: define internal swiftcc void @my_async_function2.resume.1(ptr nocapture readonly %0, ptr %1, ptr nocapture readnone %2)
; CHECK-SAME: #[[FRAMEPOINTER]]
; CHECK: tail call swiftcc void @asyncReturn({{.*}}%1)
; CHECK: ret void

define swiftcc void @top_level_caller(ptr %ctxt, ptr %task, ptr %actor) {
  %prepare = call ptr @llvm.coro.prepare.async(ptr @my_async_function)
  call swiftcc void %prepare(ptr %ctxt, ptr %task, ptr %actor)
  ret void
}

; CHECK-LABEL: define swiftcc void @top_level_caller(ptr initializes((152, 160)) %ctxt, ptr %task, ptr %actor)
; CHECK: store ptr @my_async_functionTQ0_
; CHECK: store ptr %ctxt
; CHECK: tail call swiftcc void @asyncSuspend
; CHECK: ret void

@dont_crash_on_cf_fp = constant <{ i32, i32 }>
  <{ i32 trunc ( ; Relative pointer to async function
       i64 sub (
         i64 ptrtoint (ptr @my_async_function to i64),
         i64 ptrtoint (ptr getelementptr inbounds (<{ i32, i32 }>, ptr @my_async_function_fp, i32 0, i32 1) to i64)
       )
     to i32),
     i32 128    ; Initial async context size without space for frame
}>


define swiftcc void @dont_crash_on_cf_dispatch(ptr %fnPtr, ptr %async.ctxt, ptr %task, ptr %actor) {
  %isNull = icmp eq ptr %task, null
  br i1 %isNull, label %is_null, label %is_not_null

is_null:
  ret void

is_not_null:
  tail call swiftcc void %fnPtr(ptr %async.ctxt, ptr %task, ptr %actor)
  ret void
}

define swiftcc void @dont_crash_on_cf(ptr %async.ctxt, ptr %task, ptr %actor) presplitcoroutine  {
entry:
  %id = call token @llvm.coro.id.async(i32 128, i32 16, i32 0,
          ptr @dont_crash_on_cf_fp)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr null)
  %callee_context = call ptr @llvm.coro.async.context.alloc(ptr %task, ptr @my_other_async_function_fp)
  %callee_context.return_to_caller.addr = getelementptr inbounds %async.ctxt, ptr %callee_context, i32 0, i32 1
  %resume.func_ptr = call ptr @llvm.coro.async.resume()
  store ptr %resume.func_ptr, ptr %callee_context.return_to_caller.addr
  store ptr %async.ctxt, ptr %callee_context
  %res = call {ptr, ptr, ptr} (i32, ptr, ptr, ...) @llvm.coro.suspend.async(i32 0,
                                                  ptr %resume.func_ptr,
                                                  ptr @resume_context_projection,
                                                  ptr @dont_crash_on_cf_dispatch,
                                                  ptr @asyncSuspend, ptr %callee_context, ptr %task, ptr %actor)

  call void @llvm.coro.async.context.dealloc(ptr %callee_context)
  %continuation_task_arg = extractvalue {ptr, ptr, ptr} %res, 1
  tail call swiftcc void @asyncReturn(ptr %async.ctxt, ptr %continuation_task_arg, ptr %actor)
  call i1 (ptr, i1, ...) @llvm.coro.end.async(ptr %hdl, i1 0)
  unreachable
}

@multiple_coro_end_async_fp = constant <{ i32, i32 }>
  <{ i32 trunc ( ; Relative pointer to async function
       i64 sub (
         i64 ptrtoint (ptr @multiple_coro_end_async to i64),
         i64 ptrtoint (ptr getelementptr inbounds (<{ i32, i32 }>, ptr @multiple_coro_end_async_fp, i32 0, i32 1) to i64)
       )
     to i32),
     i32 128    ; Initial async context size without space for frame
}>

define swiftcc void @must_tail_call_return(ptr %async.ctxt, ptr %task, ptr %actor) {
  musttail call swiftcc void @asyncReturn(ptr %async.ctxt, ptr %task, ptr %actor)
  ret void
}

define swiftcc void @multiple_coro_end_async(ptr %async.ctxt, ptr %task, ptr %actor) presplitcoroutine {
entry:
  %id = call token @llvm.coro.id.async(i32 128, i32 16, i32 0,
          ptr @dont_crash_on_cf_fp)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr null)
  %callee_context = call ptr @llvm.coro.async.context.alloc(ptr %task, ptr @my_other_async_function_fp)
  %callee_context.return_to_caller.addr = getelementptr inbounds %async.ctxt, ptr %callee_context, i32 0, i32 1
  %resume.func_ptr = call ptr @llvm.coro.async.resume()
  store ptr %resume.func_ptr, ptr %callee_context.return_to_caller.addr
  store ptr %async.ctxt, ptr %callee_context
  %res = call {ptr, ptr, ptr} (i32, ptr, ptr, ...) @llvm.coro.suspend.async(i32 0,
                                                  ptr %resume.func_ptr,
                                                  ptr @resume_context_projection,
                                                  ptr @dont_crash_on_cf_dispatch,
                                                  ptr @asyncSuspend, ptr %callee_context, ptr %task, ptr %actor)

  call void @llvm.coro.async.context.dealloc(ptr %callee_context)
  %continuation_task_arg = extractvalue {ptr, ptr, ptr} %res, 1
  %eq = icmp eq ptr %continuation_task_arg, null
  br i1 %eq, label %is_equal, label %is_not_equal

is_equal:
  tail call swiftcc void @asyncReturn(ptr %async.ctxt, ptr %continuation_task_arg, ptr %actor)
  call i1 (ptr, i1, ...) @llvm.coro.end.async(ptr %hdl, i1 0)
  unreachable

is_not_equal:
  call i1 (ptr, i1, ...) @llvm.coro.end.async(
                           ptr %hdl, i1 0,
                           ptr @must_tail_call_return,
                           ptr %async.ctxt, ptr %continuation_task_arg, ptr null)
  unreachable
}

; CHECK-LABEL: define internal swiftcc void @multiple_coro_end_async.resume.0(
; CHECK: musttail call swiftcc void @asyncReturn(
; CHECK: ret void

@polymorphic_suspend_return_fp = constant <{ i32, i32 }>
  <{ i32 trunc ( ; Relative pointer to async function
       i64 sub (
         i64 ptrtoint (ptr @polymorphic_suspend_return to i64),
         i64 ptrtoint (ptr getelementptr inbounds (<{ i32, i32 }>, ptr @polymorphic_suspend_return_fp, i32 0, i32 1) to i64)
       )
     to i32),
     i32 64    ; Initial async context size without space for frame
}>

define swiftcc void @polymorphic_suspend_return(ptr swiftasync %async.ctxt, ptr %task, ptr %actor) presplitcoroutine {
entry:
  %tmp = alloca { i64, i64 }, align 8
  %proj.1 = getelementptr inbounds { i64, i64 }, ptr %tmp, i64 0, i32 0
  %proj.2 = getelementptr inbounds { i64, i64 }, ptr %tmp, i64 0, i32 1

  %id = call token @llvm.coro.id.async(i32 128, i32 16, i32 0,
          ptr @polymorphic_suspend_return_fp)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr null)
  store i64 0, ptr %proj.1, align 8
  store i64 1, ptr %proj.2, align 8
  call void @some_may_write(ptr %proj.1)

	; Begin lowering: apply %my_other_async_function(%args...)

  ; setup callee context
  %callee_context = call ptr @llvm.coro.async.context.alloc(ptr %task, ptr @my_other_async_function_fp)
  ; store arguments ...
  ; ... (omitted)

  ; store the return continuation
  %callee_context.return_to_caller.addr = getelementptr inbounds %async.ctxt, ptr %callee_context, i32 0, i32 1
  %resume.func_ptr = call ptr @llvm.coro.async.resume()
  store ptr %resume.func_ptr, ptr %callee_context.return_to_caller.addr

  ; store caller context into callee context
  store ptr %async.ctxt, ptr %callee_context
  %res = call {ptr, ptr, ptr, ptr} (i32, ptr, ptr, ...)
         @llvm.coro.suspend.async.sl_p0i8p0i8p0i8p0i8s(i32 256, ;; swiftasync at 0 and swiftself at 1 in resume function
                                                 ptr %resume.func_ptr,
                                                 ptr @resume_context_projection,
                                                 ptr @my_async_function.my_other_async_function_fp.apply,
                                                 ptr @asyncSuspend, ptr %callee_context, ptr %task, ptr %actor)

  call void @llvm.coro.async.context.dealloc(ptr %callee_context)
  %continuation_task_arg = extractvalue {ptr, ptr, ptr, ptr} %res, 3
  %val = load i64, ptr %proj.1
  call void @some_user(i64 %val)
  %val.2 = load i64, ptr %proj.2
  call void @some_user(i64 %val.2)

  tail call swiftcc void @asyncReturn(ptr %async.ctxt, ptr %continuation_task_arg, ptr %actor)
  call i1 (ptr, i1, ...) @llvm.coro.end.async(ptr %hdl, i1 0)
  unreachable
}

; CHECK-LABEL: define swiftcc void @polymorphic_suspend_return(ptr swiftasync initializes((152, 160)) %async.ctxt, ptr %task, ptr %actor)
; CHECK-LABEL: define internal swiftcc void @polymorphic_suspend_return.resume.0(ptr {{.*}}swiftasync{{.*}} %0, ptr {{.*}}swiftself{{.*}} %1, ptr {{.*}}%2, ptr {{.*}}%3)
; CHECK: }

@no_coro_suspend_fp = constant <{ i32, i32 }>
  <{ i32 trunc ( ; Relative pointer to async function
       i64 sub (
         i64 ptrtoint (ptr @no_coro_suspend to i64),
         i64 ptrtoint (ptr getelementptr inbounds (<{ i32, i32 }>, ptr @no_coro_suspend_fp, i32 0, i32 1) to i64)
       )
     to i32),
     i32 128    ; Initial async context size without space for frame
}>

define swiftcc void @no_coro_suspend(ptr %async.ctx) presplitcoroutine {
entry:
  %some_alloca = alloca i64
  %id = call token @llvm.coro.id.async(i32 128, i32 16, i32 0,
          ptr @no_coro_suspend_fp)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr null)
  call void @some_may_write(ptr %some_alloca)
  call i1 (ptr, i1, ...) @llvm.coro.end.async(ptr %hdl, i1 0)
  unreachable
}

; CHECK-LABEL: define swiftcc void @no_coro_suspend
; CHECK:   [[ALLOCA:%.*]] = alloca i64
; CHECK:   call void @some_may_write(ptr {{.*}}[[ALLOCA]])

@no_coro_suspend_swifterror_fp = constant <{ i32, i32 }>
  <{ i32 trunc ( ; Relative pointer to async function
       i64 sub (
         i64 ptrtoint (ptr @no_coro_suspend_swifterror to i64),
         i64 ptrtoint (ptr getelementptr inbounds (<{ i32, i32 }>, ptr @no_coro_suspend_swifterror_fp, i32 0, i32 1) to i64)
       )
     to i32),
     i32 128    ; Initial async context size without space for frame
}>

declare void @do_with_swifterror(ptr swifterror)

define swiftcc void @no_coro_suspend_swifterror(ptr %async.ctx) presplitcoroutine {
entry:
  %some_alloca = alloca swifterror ptr
  %id = call token @llvm.coro.id.async(i32 128, i32 16, i32 0,
          ptr @no_coro_suspend_swifterror_fp)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr null)
  store ptr null, ptr %some_alloca, align 8
  call void @do_with_swifterror(ptr swifterror %some_alloca)
  call i1 (ptr, i1, ...) @llvm.coro.end.async(ptr %hdl, i1 0)
  unreachable
}

 ; CHECK-LABEL: define swiftcc void @no_coro_suspend_swifterror
 ; CHECK:  [[ALLOCA:%.*]] = alloca swifterror ptr
 ; CHECK:   store ptr null, ptr [[ALLOCA]]
 ; CHECK:   call void @do_with_swifterror(ptr {{.*}}swifterror{{.*}} [[ALLOCA]])

@undefined_coro_async_resume_fp = constant <{ i32, i32 }>
  <{ i32 trunc (
       i64 sub (
         i64 ptrtoint (ptr @undefined_coro_async_resume to i64),
         i64 ptrtoint (ptr getelementptr inbounds (<{ i32, i32 }>, ptr @undefined_coro_async_resume_fp, i32 0, i32 1) to i64)
       )
     to i32),
     i32 24
}>

declare void @crash()
declare void @use(ptr)

define swiftcc void @undefined_coro_async_resume(ptr %async.ctx) presplitcoroutine {
entry:
  %id = call token @llvm.coro.id.async(i32 24, i32 16, i32 0, ptr @undefined_coro_async_resume_fp)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr null)
  %undefined_resume_pointer = call ptr @llvm.coro.async.resume()
  call void @use(ptr %undefined_resume_pointer)
  call void @crash()
  %unused = call i1 (ptr, i1, ...) @llvm.coro.end.async(ptr %hdl, i1 false)
  unreachable
}
; CHECK-LABEL: define swiftcc void @undefined_coro_async_resume
; CHECK-NOT: @llvm.coro.async.resume
; CHECK: call void @use(ptr null)
; CHECK: ret

declare { ptr, ptr, ptr, ptr } @llvm.coro.suspend.async.sl_p0i8p0i8p0i8p0i8s(i32, ptr, ptr, ...)
declare ptr @llvm.coro.prepare.async(ptr)
declare token @llvm.coro.id.async(i32, i32, i32, ptr)
declare ptr @llvm.coro.begin(token, ptr)
declare i1 @llvm.coro.end.async(ptr, i1, ...)
declare i1 @llvm.coro.end(ptr, i1, token)
declare {ptr, ptr, ptr} @llvm.coro.suspend.async(i32, ptr, ptr, ...)
declare ptr @llvm.coro.async.context.alloc(ptr, ptr)
declare void @llvm.coro.async.context.dealloc(ptr)
declare swiftcc void @asyncReturn(ptr, ptr, ptr)
declare swiftcc void @asyncSuspend(ptr, ptr, ptr)
declare ptr @llvm.coro.async.resume()
declare void @llvm.coro.async.size.replace(ptr, ptr)
declare ptr @hide(ptr)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
; CHECK: ![[SP1]] = distinct !DISubprogram(name: "my_async_function",
; CHECK-SAME:                              linkageName: "my_async_function",
; CHECK-SAME:                              scopeLine: 1
!1 = distinct !DISubprogram(name: "my_async_function",
                            linkageName: "my_async_function",
                            scope: !2, file: !3, line: 1, type: !4,
                            scopeLine: 1, spFlags: DISPFlagDefinition, unit: !2)
; CHECK: ![[SP2]] = distinct !DISubprogram(name: "my_async_function",
; CHECK-SAME:                              linkageName: "my_async_functionTQ0_",
; CHECK-SAME:                              scopeLine: 2
!2 = distinct !DICompileUnit(language: DW_LANG_Swift, file: !3, emissionKind: FullDebug)
!3 = !DIFile(filename: "/tmp/1.swift", directory: "/")
!4 = !DISubroutineType(types: !{})
!5 = !DILocation(line: 2, column: 0, scope: !1)

; CHECK: ![[SP3]] = distinct !DISubprogram(name: "my_async_function2",
; CHECK-SAME:                              linkageName: "my_async_function2",
; CHECK-SAME:                              scopeLine: 1
!6 = distinct !DISubprogram(name: "my_async_function2",
                            linkageName: "my_async_function2",
                            scope: !2, file: !3, line: 1, type: !4,
                            scopeLine: 1, spFlags: DISPFlagDefinition, unit: !2)
; CHECK: ![[SP4]] = distinct !DISubprogram(name: "my_async_function2",
; CHECK-SAME:                              linkageName: "my_async_function2.resume.0",
; CHECK-SAME:                              scopeLine: 1
!7 = !DILexicalBlockFile(scope: !6, file: !8, discriminator: 0)
!8 = !DIFile(filename: "/tmp/fake.cpp", directory: "/")
!9 = !DILocation(line: 2, column: 0, scope: !7)
