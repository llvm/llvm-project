; RUN: opt < %s -enable-coroutines -passes='default<O2>' -S | FileCheck --check-prefixes=CHECK %s
; RUN: opt < %s -enable-coroutines -O0 -S | FileCheck --check-prefixes=CHECK-O0 %s
target datalayout = "p:64:64:64"

%async.task = type { i64 }
%async.actor = type { i64 }
%async.fp = type <{ i32, i32 }>

%async.ctxt = type { i8*, void (i8*, %async.task*, %async.actor*)* }

; The async callee.
@my_other_async_function_fp = external global <{ i32, i32 }>
declare void @my_other_async_function(i8* %async.ctxt)

; The current async function (the caller).
; This struct describes an async function. The first field is the
; relative offset to the async function implementation, the second field is the
; size needed for the async context of the current async function.

@my_async_function_fp = constant <{ i32, i32 }>
  <{ i32 trunc ( ; Relative pointer to async function
       i64 sub (
         i64 ptrtoint (void (i8*, %async.task*, %async.actor*)* @my_async_function to i64),
         i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32 }>, <{ i32, i32 }>* @my_async_function_fp, i32 0, i32 1) to i64)
       )
     to i32),
     i32 128    ; Initial async context size without space for frame
}>
@my_async_function_pa_fp = constant <{ i32, i32 }>
  <{ i32 trunc (
       i64 sub (
         i64 ptrtoint (void (i8*, %async.task*, %async.actor*)* @my_async_function_pa to i64),
         i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32 }>, <{ i32, i32 }>* @my_async_function_pa_fp, i32 0, i32 1) to i64)
       )
     to i32),
     i32 8
}>

; Function that implements the dispatch to the callee function.
define swiftcc void @my_async_function.my_other_async_function_fp.apply(i8* %fnPtr, i8* %async.ctxt, %async.task* %task, %async.actor* %actor) {
  %callee = bitcast i8* %fnPtr to void(i8*, %async.task*, %async.actor*)*
  tail call swiftcc void %callee(i8* %async.ctxt, %async.task* %task, %async.actor* %actor)
  ret void
}

declare void @some_user(i64)
declare void @some_may_write(i64*)

define i8* @__swift_async_resume_project_context(i8* %ctxt) {
entry:
  %resume_ctxt_addr = bitcast i8* %ctxt to i8**
  %resume_ctxt = load i8*, i8** %resume_ctxt_addr, align 8
  ret i8* %resume_ctxt
}

define i8* @resume_context_projection(i8* %ctxt) {
entry:
  %resume_ctxt_addr = bitcast i8* %ctxt to i8**
  %resume_ctxt = load i8*, i8** %resume_ctxt_addr, align 8
  ret i8* %resume_ctxt
}


define swiftcc void @my_async_function(i8* swiftasync %async.ctxt, %async.task* %task, %async.actor* %actor) "coroutine.presplit"="1" !dbg !1 {
entry:
  %tmp = alloca { i64, i64 }, align 8
  %vector = alloca <4 x double>, align 16
  %proj.1 = getelementptr inbounds { i64, i64 }, { i64, i64 }* %tmp, i64 0, i32 0
  %proj.2 = getelementptr inbounds { i64, i64 }, { i64, i64 }* %tmp, i64 0, i32 1

  %id = call token @llvm.coro.id.async(i32 128, i32 16, i32 0,
          i8* bitcast (<{i32, i32}>* @my_async_function_fp to i8*))
  %hdl = call i8* @llvm.coro.begin(token %id, i8* null)
  store i64 0, i64* %proj.1, align 8
  store i64 1, i64* %proj.2, align 8
  call void @some_may_write(i64* %proj.1)

	; Begin lowering: apply %my_other_async_function(%args...)

  ; setup callee context
  %arg0 = bitcast %async.task* %task to i8*
  %arg1 = bitcast <{ i32, i32}>* @my_other_async_function_fp to i8*
  %callee_context = call i8* @llvm.coro.async.context.alloc(i8* %arg0, i8* %arg1)
	%callee_context.0 = bitcast i8* %callee_context to %async.ctxt*
  ; store arguments ...
  ; ... (omitted)

  ; store the return continuation
  %callee_context.return_to_caller.addr = getelementptr inbounds %async.ctxt, %async.ctxt* %callee_context.0, i32 0, i32 1
  %return_to_caller.addr = bitcast void(i8*, %async.task*, %async.actor*)** %callee_context.return_to_caller.addr to i8**
  %resume.func_ptr = call i8* @llvm.coro.async.resume()
  store i8* %resume.func_ptr, i8** %return_to_caller.addr

  ; store caller context into callee context
  %callee_context.caller_context.addr = getelementptr inbounds %async.ctxt, %async.ctxt* %callee_context.0, i32 0, i32 0
  store i8* %async.ctxt, i8** %callee_context.caller_context.addr
  %resume_proj_fun = bitcast i8*(i8*)* @__swift_async_resume_project_context to i8*
  %callee = bitcast void(i8*, %async.task*, %async.actor*)* @asyncSuspend to i8*
  %vector_spill = load <4 x double>, <4 x double>* %vector, align 16
  %res = call {i8*, i8*, i8*} (i32, i8*, i8*, ...) @llvm.coro.suspend.async(i32 0,
                                                  i8* %resume.func_ptr,
                                                  i8* %resume_proj_fun,
                                                  void (i8*, i8*, %async.task*, %async.actor*)* @my_async_function.my_other_async_function_fp.apply,
                                                  i8* %callee, i8* %callee_context, %async.task* %task, %async.actor *%actor), !dbg !5

  call void @llvm.coro.async.context.dealloc(i8* %callee_context)
  %continuation_task_arg = extractvalue {i8*, i8*, i8*} %res, 1
  %task.2 =  bitcast i8* %continuation_task_arg to %async.task*
  %val = load i64, i64* %proj.1
  call void @some_user(i64 %val)
  %val.2 = load i64, i64* %proj.2
  call void @some_user(i64 %val.2)
  store <4 x double> %vector_spill, <4 x double>* %vector, align 16
  tail call swiftcc void @asyncReturn(i8* %async.ctxt, %async.task* %task.2, %async.actor* %actor)
  call i1 (i8*, i1, ...) @llvm.coro.end.async(i8* %hdl, i1 0)
  unreachable
}

define void @my_async_function_pa(i8* %ctxt, %async.task* %task, %async.actor* %actor) {
  call void @llvm.coro.async.size.replace(i8* bitcast (<{i32, i32}>* @my_async_function_pa_fp to i8*), i8* bitcast (<{i32, i32}>* @my_async_function_fp to i8*))
  call swiftcc void @my_async_function(i8* %ctxt, %async.task* %task, %async.actor* %actor)
  ret void
}

; Make sure we update the async function pointer
; CHECK: @my_async_function_fp = constant <{ i32, i32 }> <{ {{.*}}, i32 176 }
; CHECK: @my_async_function_pa_fp = constant <{ i32, i32 }> <{ {{.*}}, i32 176 }
; CHECK: @my_async_function2_fp = constant <{ i32, i32 }> <{ {{.*}}, i32 176 }

; CHECK-LABEL: define swiftcc void @my_async_function(i8* swiftasync %async.ctxt, %async.task* %task, %async.actor* %actor)
; CHECK-O0-LABEL: define swiftcc void @my_async_function(i8* swiftasync %async.ctxt, %async.task* %task, %async.actor* %actor)
; CHECK-SAME: !dbg ![[SP1:[0-9]+]] {
; CHECK: coro.return:
; CHECK:   [[FRAMEPTR:%.*]] = getelementptr inbounds i8, i8* %async.ctxt, i64 128
; CHECK:   [[ACTOR_SPILL_ADDR:%.*]] = getelementptr inbounds i8, i8* %async.ctxt, i64 152
; CHECK:   [[CAST1:%.*]] = bitcast i8* [[ACTOR_SPILL_ADDR]] to %async.actor**
; CHECK:   store %async.actor* %actor, %async.actor** [[CAST1]]
; CHECK:   [[ADDR1:%.*]]  = getelementptr inbounds i8, i8* %async.ctxt, i64 144
; CHECK:   [[ASYNC_CTXT_SPILL_ADDR:%.*]] = bitcast i8* [[ADDR1]] to i8**
; CHECK:   store i8* %async.ctxt, i8** [[ASYNC_CTXT_SPILL_ADDR]]
; CHECK:   [[ALLOCA_PRJ1:%.*]] = bitcast i8* [[FRAMEPTR]] to i64*
; CHECK:   [[ALLOCA_PRJ2:%.*]] = getelementptr inbounds i8, i8* %async.ctxt, i64 136
; CHECK:   [[ADDR2:%.*]] = bitcast i8* [[ALLOCA_PRJ2]] to i64*
; CHECK:   store i64 0, i64* [[ALLOCA_PRJ1]]
; CHECK:   store i64 1, i64* [[ADDR2]]
; CHECK:   tail call void @some_may_write(i64* nonnull %proj.1)
; CHECK:   [[TASK:%.*]] = bitcast %async.task* %task to i8*
; CHECK:   [[CALLEE_CTXT:%.*]] = tail call i8* @llvm.coro.async.context.alloc(i8* [[TASK]], i8* bitcast (<{ i32, i32 }>* @my_other_async_function_fp to i8*))
; CHECK:   [[CALLEE_CTXT_SPILL:%.*]] = getelementptr inbounds i8, i8* %async.ctxt, i64 160
; CHECK:   [[CAST2:%.*]] = bitcast i8* [[CALLEE_CTXT_SPILL]] to i8**
; CHECK:   store i8* [[CALLEE_CTXT]], i8** [[CAST2]]
; CHECK:   [[TYPED_RETURN_TO_CALLER_ADDR:%.*]] = getelementptr inbounds i8, i8* [[CALLEE_CTXT]], i64 8
; CHECK:   [[RETURN_TO_CALLER_ADDR:%.*]] = bitcast i8* [[TYPED_RETURN_TO_CALLER_ADDR]] to i8**
; CHECK:   store i8* bitcast (void (i8*, i8*, i8*)* @my_async_functionTQ0_ to i8*), i8** [[RETURN_TO_CALLER_ADDR]]
; CHECK:   [[CALLER_CONTEXT_ADDR:%.*]] = bitcast i8* [[CALLEE_CTXT]] to i8**
; CHECK:   store i8* %async.ctxt, i8** [[CALLER_CONTEXT_ADDR]]
; Make sure the spill is underaligned to the max context alignment (16).
; CHECK-O0:   [[VECTOR_SPILL:%.*]] = load <4 x double>, <4 x double>* {{.*}}
; CHECK-O0:   [[VECTOR_SPILL_ADDR:%.*]] = getelementptr inbounds %my_async_function.Frame, %my_async_function.Frame* {{.*}}, i32 0, i32 1
; CHECK-O0:   store <4 x double> [[VECTOR_SPILL]], <4 x double>* [[VECTOR_SPILL_ADDR]], align 16
; CHECK:   tail call swiftcc void @asyncSuspend(i8* [[CALLEE_CTXT]], %async.task* %task, %async.actor* %actor)
; CHECK:   ret void
; CHECK: }

; CHECK-LABEL: define internal swiftcc void @my_async_functionTQ0_(i8* nocapture readonly swiftasync %0, i8* %1, i8* nocapture readnone %2)
; CHECK-O0-LABEL: define internal swiftcc void @my_async_functionTQ0_(i8* swiftasync %0, i8* %1, i8* %2)
; CHECK-SAME: !dbg ![[SP2:[0-9]+]] {
; CHECK: entryresume.0:
; CHECK:   [[CALLER_CONTEXT_ADDR:%.*]] = bitcast i8* %0 to i8**
; CHECK:   [[CALLER_CONTEXT:%.*]] = load i8*, i8** [[CALLER_CONTEXT_ADDR]]
; CHECK:   [[FRAME_PTR:%.*]] = getelementptr inbounds i8, i8* [[CALLER_CONTEXT]], i64 128
; CHECK-O0:   [[VECTOR_SPILL_ADDR:%.*]] = getelementptr inbounds %my_async_function.Frame, %my_async_function.Frame* {{.*}}, i32 0, i32 1
; CHECK-O0:   load <4 x double>, <4 x double>* [[VECTOR_SPILL_ADDR]], align 16
; CHECK:   [[CALLEE_CTXT_SPILL_ADDR:%.*]] = getelementptr inbounds i8, i8* [[CALLER_CONTEXT]], i64 160
; CHECK:   [[CAST1:%.*]] = bitcast i8* [[CALLEE_CTXT_SPILL_ADDR]] to i8**
; CHECK:   [[CALLEE_CTXT_RELOAD:%.*]] = load i8*, i8** [[CAST1]]
; CHECK:   [[ACTOR_RELOAD_ADDR:%.*]] = getelementptr inbounds i8, i8* [[CALLER_CONTEXT]], i64 152
; CHECK:   [[CAST2:%.*]] = bitcast i8* [[ACTOR_RELOAD_ADDR]] to %async.actor**
; CHECK:   [[ACTOR_RELOAD:%.*]] = load %async.actor*, %async.actor** [[CAST2]]
; CHECK:   [[ADDR1:%.*]] = getelementptr inbounds i8, i8* [[CALLER_CONTEXT]], i64 144
; CHECK:   [[ASYNC_CTXT_RELOAD_ADDR:%.*]] = bitcast i8* [[ADDR1]] to i8**
; CHECK:   [[ASYNC_CTXT_RELOAD:%.*]] = load i8*, i8** [[ASYNC_CTXT_RELOAD_ADDR]]
; CHECK:   [[ALLOCA_PRJ2:%.*]] = getelementptr inbounds i8, i8* [[CALLER_CONTEXT]], i64 136
; CHECK:   [[ADDR2:%.*]] = bitcast i8* [[ALLOCA_PRJ2]] to i64*
; CHECK:   [[ALLOCA_PRJ1:%.*]] = bitcast i8* [[FRAME_PTR]] to i64*
; CHECK:   tail call void @llvm.coro.async.context.dealloc(i8* [[CALLEE_CTXT_RELOAD]])
; CHECK:   [[TASK_ARG:%.*]] = bitcast i8* %1 to %async.task*
; CHECK:   [[VAL1:%.*]] = load i64, i64* [[ALLOCA_PRJ1]]
; CHECK:   tail call void @some_user(i64 [[VAL1]])
; CHECK:   [[VAL2:%.*]] = load i64, i64* [[ADDR2]]
; CHECK:   tail call void @some_user(i64 [[VAL2]])
; CHECK:   tail call swiftcc void @asyncReturn(i8* [[ASYNC_CTXT_RELOAD]], %async.task* [[TASK_ARG]], %async.actor* [[ACTOR_RELOAD]])
; CHECK:   ret void
; CHECK: }

@my_async_function2_fp = constant <{ i32, i32 }>
  <{ i32 trunc ( ; Relative pointer to async function
       i64 sub (
         i64 ptrtoint (void (%async.task*, %async.actor*, i8*)* @my_async_function2 to i64),
         i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32 }>, <{ i32, i32 }>* @my_async_function2_fp, i32 0, i32 1) to i64)
       )
     to i32),
     i32 128    ; Initial async context size without space for frame
  }>

define swiftcc void @my_async_function2(%async.task* %task, %async.actor* %actor, i8* %async.ctxt) "coroutine.presplit"="1" "frame-pointer"="all" !dbg !6 {
entry:

  %id = call token @llvm.coro.id.async(i32 128, i32 16, i32 2, i8* bitcast (<{i32, i32}>* @my_async_function2_fp to i8*))
  %hdl = call i8* @llvm.coro.begin(token %id, i8* null)
  ; setup callee context
  %arg0 = bitcast %async.task* %task to i8*
  %arg1 = bitcast <{ i32, i32}>* @my_other_async_function_fp to i8*
  %callee_context = call i8* @llvm.coro.async.context.alloc(i8* %arg0, i8* %arg1)

	%callee_context.0 = bitcast i8* %callee_context to %async.ctxt*
  %callee_context.return_to_caller.addr = getelementptr inbounds %async.ctxt, %async.ctxt* %callee_context.0, i32 0, i32 1
  %return_to_caller.addr = bitcast void(i8*, %async.task*, %async.actor*)** %callee_context.return_to_caller.addr to i8**
  %resume.func_ptr = call i8* @llvm.coro.async.resume()
  store i8* %resume.func_ptr, i8** %return_to_caller.addr
  %callee_context.caller_context.addr = getelementptr inbounds %async.ctxt, %async.ctxt* %callee_context.0, i32 0, i32 0
  store i8* %async.ctxt, i8** %callee_context.caller_context.addr
  %resume_proj_fun = bitcast i8*(i8*)* @resume_context_projection to i8*
  %callee = bitcast void(i8*, %async.task*, %async.actor*)* @asyncSuspend to i8*
  %task.casted = bitcast i8* %arg0 to %async.task*
  %res = call {i8*, i8*, i8*} (i32, i8*, i8*, ...) @llvm.coro.suspend.async(i32 2,
                                                  i8* %resume.func_ptr,
                                                  i8* %resume_proj_fun,
                                                  void (i8*, i8*, %async.task*, %async.actor*)* @my_async_function.my_other_async_function_fp.apply,
                                                  i8* %callee, i8* %callee_context, %async.task* %task.casted, %async.actor *%actor), !dbg !9

  %continuation_task_arg = extractvalue {i8*, i8*, i8*} %res, 0
  %task.2 =  bitcast i8* %continuation_task_arg to %async.task*

	%callee_context.0.1 = bitcast i8* %callee_context to %async.ctxt*
  %callee_context.return_to_caller.addr.1 = getelementptr inbounds %async.ctxt, %async.ctxt* %callee_context.0.1, i32 0, i32 1
  %return_to_caller.addr.1 = bitcast void(i8*, %async.task*, %async.actor*)** %callee_context.return_to_caller.addr.1 to i8**
  %resume.func_ptr.1 = call i8* @llvm.coro.async.resume()
  store i8* %resume.func_ptr.1, i8** %return_to_caller.addr.1
  %callee_context.caller_context.addr.1 = getelementptr inbounds %async.ctxt, %async.ctxt* %callee_context.0.1, i32 0, i32 0
  store i8* %async.ctxt, i8** %callee_context.caller_context.addr.1
  %resume_proj_fun.2 = bitcast i8*(i8*)* @resume_context_projection to i8*
  %callee.2 = bitcast void(i8*, %async.task*, %async.actor*)* @asyncSuspend to i8*
  %res.2 = call {i8*, i8*, i8*} (i32, i8*, i8*, ...) @llvm.coro.suspend.async(i32 0,
                                                  i8* %resume.func_ptr.1,
                                                  i8* %resume_proj_fun.2,
                                                  void (i8*, i8*, %async.task*, %async.actor*)* @my_async_function.my_other_async_function_fp.apply,
                                                  i8* %callee.2, i8* %callee_context, %async.task* %task.casted, %async.actor *%actor)

  call void @llvm.coro.async.context.dealloc(i8* %callee_context)
  %continuation_actor_arg = extractvalue {i8*, i8*, i8*} %res.2, 1
  %actor.2 =  bitcast i8* %continuation_actor_arg to %async.actor*

  tail call swiftcc void @asyncReturn(i8* %async.ctxt, %async.task* %task.2, %async.actor* %actor.2)
  call i1 @llvm.coro.end(i8* %hdl, i1 0)
  unreachable
}

; CHECK-LABEL: define swiftcc void @my_async_function2(%async.task* %task, %async.actor* %actor, i8* %async.ctxt)
; CHECK-SAME: #[[FRAMEPOINTER:[0-9]+]]
; CHECK-SAME: !dbg ![[SP3:[0-9]+]]
; CHECK: store i8* %async.ctxt,
; CHECK: store %async.actor* %actor,
; CHECK: store %async.task* %task,
; CHECK: [[CALLEE_CTXT:%.*]] =  tail call i8* @llvm.coro.async.context.alloc(
; CHECK: store i8* [[CALLEE_CTXT]],
; CHECK: store i8* bitcast (void (i8*, i8*, i8*)* @my_async_function2.resume.0 to i8*),
; CHECK: store i8* %async.ctxt,
; CHECK: tail call swiftcc void @asyncSuspend(i8* [[CALLEE_CTXT]], %async.task* %task, %async.actor* %actor)
; CHECK: ret void

; CHECK-LABEL: define internal swiftcc void @my_async_function2.resume.0(i8* %0, i8* nocapture readnone %1, i8* nocapture readonly %2)
; CHECK-SAME: #[[FRAMEPOINTER]]
; CHECK-SAME: !dbg ![[SP4:[0-9]+]]
; CHECK: [[CALLEE_CTXT_ADDR:%.*]] = bitcast i8* %2 to i8**
; CHECK: [[CALLEE_CTXT:%.*]] = load i8*, i8** [[CALLEE_CTXT_ADDR]]
; CHECK: [[CALLEE_CTXT_SPILL_ADDR:%.*]] = getelementptr inbounds i8, i8* [[CALLEE_CTXT]], i64 152
; CHECK: [[CALLEE_CTXT_SPILL_ADDR2:%.*]] = bitcast i8* [[CALLEE_CTXT_SPILL_ADDR]] to i8**
; CHECK: store i8* bitcast (void (i8*, i8*, i8*)* @my_async_function2.resume.1 to i8*),
; CHECK: [[CALLLE_CTXT_RELOAD:%.*]] = load i8*, i8** [[CALLEE_CTXT_SPILL_ADDR2]]
; CHECK: tail call swiftcc void @asyncSuspend(i8* [[CALLEE_CTXT_RELOAD]]
; CHECK: ret void

; CHECK-LABEL: define internal swiftcc void @my_async_function2.resume.1(i8* nocapture readonly %0, i8* %1, i8* nocapture readnone %2)
; CHECK-SAME: #[[FRAMEPOINTER]]
; CHECK: bitcast i8* %0 to i8**
; CHECK: [[ACTOR_ARG:%.*]] = bitcast i8* %1
; CHECK: tail call swiftcc void @asyncReturn({{.*}}[[ACTOR_ARG]])
; CHECK: ret void

define swiftcc void @top_level_caller(i8* %ctxt, i8* %task, i8* %actor) {
  %prepare = call i8* @llvm.coro.prepare.async(i8* bitcast (void (i8*, %async.task*,  %async.actor*)* @my_async_function to i8*))
  %f = bitcast i8* %prepare to void (i8*, i8*, i8*)*
  call swiftcc void %f(i8* %ctxt, i8* %task, i8* %actor)
  ret void
}

; CHECK-LABEL: define swiftcc void @top_level_caller(i8* %ctxt, i8* %task, i8* %actor)
; CHECK: store i8* bitcast (void (i8*, i8*, i8*)* @my_async_functionTQ0_
; CHECK: store i8* %ctxt
; CHECK: tail call swiftcc void @asyncSuspend
; CHECK: ret void

@dont_crash_on_cf_fp = constant <{ i32, i32 }>
  <{ i32 trunc ( ; Relative pointer to async function
       i64 sub (
         i64 ptrtoint (void (i8*, %async.task*, %async.actor*)* @my_async_function to i64),
         i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32 }>, <{ i32, i32 }>* @my_async_function_fp, i32 0, i32 1) to i64)
       )
     to i32),
     i32 128    ; Initial async context size without space for frame
}>


define swiftcc void @dont_crash_on_cf_dispatch(i8* %fnPtr, i8* %async.ctxt, %async.task* %task, %async.actor* %actor) {
  %isNull = icmp eq %async.task* %task, null
  br i1 %isNull, label %is_null, label %is_not_null

is_null:
  ret void

is_not_null:
  %callee = bitcast i8* %fnPtr to void(i8*, %async.task*, %async.actor*)*
  tail call swiftcc void %callee(i8* %async.ctxt, %async.task* %task, %async.actor* %actor)
  ret void
}

define swiftcc void @dont_crash_on_cf(i8* %async.ctxt, %async.task* %task, %async.actor* %actor) "coroutine.presplit"="1"  {
entry:
  %id = call token @llvm.coro.id.async(i32 128, i32 16, i32 0,
          i8* bitcast (<{i32, i32}>* @dont_crash_on_cf_fp to i8*))
  %hdl = call i8* @llvm.coro.begin(token %id, i8* null)
  %arg0 = bitcast %async.task* %task to i8*
  %arg1 = bitcast <{ i32, i32}>* @my_other_async_function_fp to i8*
  %callee_context = call i8* @llvm.coro.async.context.alloc(i8* %arg0, i8* %arg1)
	%callee_context.0 = bitcast i8* %callee_context to %async.ctxt*
  %callee_context.return_to_caller.addr = getelementptr inbounds %async.ctxt, %async.ctxt* %callee_context.0, i32 0, i32 1
  %return_to_caller.addr = bitcast void(i8*, %async.task*, %async.actor*)** %callee_context.return_to_caller.addr to i8**
  %resume.func_ptr = call i8* @llvm.coro.async.resume()
  store i8* %resume.func_ptr, i8** %return_to_caller.addr
  %callee_context.caller_context.addr = getelementptr inbounds %async.ctxt, %async.ctxt* %callee_context.0, i32 0, i32 0
  store i8* %async.ctxt, i8** %callee_context.caller_context.addr
  %resume_proj_fun = bitcast i8*(i8*)* @resume_context_projection to i8*
  %callee = bitcast void(i8*, %async.task*, %async.actor*)* @asyncSuspend to i8*
  %res = call {i8*, i8*, i8*} (i32, i8*, i8*, ...) @llvm.coro.suspend.async(i32 0,
                                                  i8* %resume.func_ptr,
                                                  i8* %resume_proj_fun,
                                                  void (i8*, i8*, %async.task*, %async.actor*)* @dont_crash_on_cf_dispatch,
                                                  i8* %callee, i8* %callee_context, %async.task* %task, %async.actor *%actor)

  call void @llvm.coro.async.context.dealloc(i8* %callee_context)
  %continuation_task_arg = extractvalue {i8*, i8*, i8*} %res, 1
  %task.2 =  bitcast i8* %continuation_task_arg to %async.task*
  tail call swiftcc void @asyncReturn(i8* %async.ctxt, %async.task* %task.2, %async.actor* %actor)
  call i1 (i8*, i1, ...) @llvm.coro.end.async(i8* %hdl, i1 0)
  unreachable
}

@multiple_coro_end_async_fp = constant <{ i32, i32 }>
  <{ i32 trunc ( ; Relative pointer to async function
       i64 sub (
         i64 ptrtoint (void (i8*, %async.task*, %async.actor*)* @multiple_coro_end_async to i64),
         i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32 }>, <{ i32, i32 }>* @multiple_coro_end_async_fp, i32 0, i32 1) to i64)
       )
     to i32),
     i32 128    ; Initial async context size without space for frame
}>

define swiftcc void @must_tail_call_return(i8* %async.ctxt, %async.task* %task, %async.actor* %actor) {
  musttail call swiftcc void @asyncReturn(i8* %async.ctxt, %async.task* %task, %async.actor* %actor)
  ret void
}

define swiftcc void @multiple_coro_end_async(i8* %async.ctxt, %async.task* %task, %async.actor* %actor) "coroutine.presplit"="1" {
entry:
  %id = call token @llvm.coro.id.async(i32 128, i32 16, i32 0,
          i8* bitcast (<{i32, i32}>* @dont_crash_on_cf_fp to i8*))
  %hdl = call i8* @llvm.coro.begin(token %id, i8* null)
  %arg0 = bitcast %async.task* %task to i8*
  %arg1 = bitcast <{ i32, i32}>* @my_other_async_function_fp to i8*
  %callee_context = call i8* @llvm.coro.async.context.alloc(i8* %arg0, i8* %arg1)
	%callee_context.0 = bitcast i8* %callee_context to %async.ctxt*
  %callee_context.return_to_caller.addr = getelementptr inbounds %async.ctxt, %async.ctxt* %callee_context.0, i32 0, i32 1
  %return_to_caller.addr = bitcast void(i8*, %async.task*, %async.actor*)** %callee_context.return_to_caller.addr to i8**
  %resume.func_ptr = call i8* @llvm.coro.async.resume()
  store i8* %resume.func_ptr, i8** %return_to_caller.addr
  %callee_context.caller_context.addr = getelementptr inbounds %async.ctxt, %async.ctxt* %callee_context.0, i32 0, i32 0
  store i8* %async.ctxt, i8** %callee_context.caller_context.addr
  %resume_proj_fun = bitcast i8*(i8*)* @resume_context_projection to i8*
  %callee = bitcast void(i8*, %async.task*, %async.actor*)* @asyncSuspend to i8*
  %res = call {i8*, i8*, i8*} (i32, i8*, i8*, ...) @llvm.coro.suspend.async(i32 0,
                                                  i8* %resume.func_ptr,
                                                  i8* %resume_proj_fun,
                                                  void (i8*, i8*, %async.task*, %async.actor*)* @dont_crash_on_cf_dispatch,
                                                  i8* %callee, i8* %callee_context, %async.task* %task, %async.actor *%actor)

  call void @llvm.coro.async.context.dealloc(i8* %callee_context)
  %continuation_task_arg = extractvalue {i8*, i8*, i8*} %res, 1
  %task.2 =  bitcast i8* %continuation_task_arg to %async.task*
  %eq = icmp eq i8 * %continuation_task_arg, null
  br i1 %eq, label %is_equal, label %is_not_equal

is_equal:
  tail call swiftcc void @asyncReturn(i8* %async.ctxt, %async.task* %task.2, %async.actor* %actor)
  call i1 (i8*, i1, ...) @llvm.coro.end.async(i8* %hdl, i1 0)
  unreachable

is_not_equal:
  call i1 (i8*, i1, ...) @llvm.coro.end.async(
                           i8* %hdl, i1 0,
                           void (i8*, %async.task*, %async.actor*)* @must_tail_call_return,
                           i8* %async.ctxt, %async.task* %task.2, %async.actor* null)
  unreachable
}

; CHECK-LABEL: define internal swiftcc void @multiple_coro_end_async.resume.0(
; CHECK: musttail call swiftcc void @asyncReturn(
; CHECK: ret void

@polymorphic_suspend_return_fp = constant <{ i32, i32 }>
  <{ i32 trunc ( ; Relative pointer to async function
       i64 sub (
         i64 ptrtoint (void (i8*, %async.task*, %async.actor*)* @polymorphic_suspend_return to i64),
         i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32 }>, <{ i32, i32 }>* @polymorphic_suspend_return_fp, i32 0, i32 1) to i64)
       )
     to i32),
     i32 64    ; Initial async context size without space for frame
}>

define swiftcc void @polymorphic_suspend_return(i8* swiftasync %async.ctxt, %async.task* %task, %async.actor* %actor) "coroutine.presplit"="1" {
entry:
  %tmp = alloca { i64, i64 }, align 8
  %proj.1 = getelementptr inbounds { i64, i64 }, { i64, i64 }* %tmp, i64 0, i32 0
  %proj.2 = getelementptr inbounds { i64, i64 }, { i64, i64 }* %tmp, i64 0, i32 1

  %id = call token @llvm.coro.id.async(i32 128, i32 16, i32 0,
          i8* bitcast (<{i32, i32}>* @polymorphic_suspend_return_fp to i8*))
  %hdl = call i8* @llvm.coro.begin(token %id, i8* null)
  store i64 0, i64* %proj.1, align 8
  store i64 1, i64* %proj.2, align 8
  call void @some_may_write(i64* %proj.1)

	; Begin lowering: apply %my_other_async_function(%args...)

  ; setup callee context
  %arg0 = bitcast %async.task* %task to i8*
  %arg1 = bitcast <{ i32, i32}>* @my_other_async_function_fp to i8*
  %callee_context = call i8* @llvm.coro.async.context.alloc(i8* %arg0, i8* %arg1)
	%callee_context.0 = bitcast i8* %callee_context to %async.ctxt*
  ; store arguments ...
  ; ... (omitted)

  ; store the return continuation
  %callee_context.return_to_caller.addr = getelementptr inbounds %async.ctxt, %async.ctxt* %callee_context.0, i32 0, i32 1
  %return_to_caller.addr = bitcast void(i8*, %async.task*, %async.actor*)** %callee_context.return_to_caller.addr to i8**
  %resume.func_ptr = call i8* @llvm.coro.async.resume()
  store i8* %resume.func_ptr, i8** %return_to_caller.addr

  ; store caller context into callee context
  %callee_context.caller_context.addr = getelementptr inbounds %async.ctxt, %async.ctxt* %callee_context.0, i32 0, i32 0
  store i8* %async.ctxt, i8** %callee_context.caller_context.addr
  %resume_proj_fun = bitcast i8*(i8*)* @resume_context_projection to i8*
  %callee = bitcast void(i8*, %async.task*, %async.actor*)* @asyncSuspend to i8*
  %res = call {i8*, i8*, i8*, i8*} (i32, i8*, i8*, ...)
         @llvm.coro.suspend.async.sl_p0i8p0i8p0i8p0i8s(i32 256, ;; swiftasync at 0 and swiftself at 1 in resume function
                                                 i8* %resume.func_ptr,
                                                 i8* %resume_proj_fun,
                                                 void (i8*, i8*, %async.task*, %async.actor*)* @my_async_function.my_other_async_function_fp.apply,
                                                 i8* %callee, i8* %callee_context, %async.task* %task, %async.actor *%actor)

  call void @llvm.coro.async.context.dealloc(i8* %callee_context)
  %continuation_task_arg = extractvalue {i8*, i8*, i8*, i8*} %res, 3
  %task.2 =  bitcast i8* %continuation_task_arg to %async.task*
  %val = load i64, i64* %proj.1
  call void @some_user(i64 %val)
  %val.2 = load i64, i64* %proj.2
  call void @some_user(i64 %val.2)

  tail call swiftcc void @asyncReturn(i8* %async.ctxt, %async.task* %task.2, %async.actor* %actor)
  call i1 (i8*, i1, ...) @llvm.coro.end.async(i8* %hdl, i1 0)
  unreachable
}

; CHECK-LABEL: define swiftcc void @polymorphic_suspend_return(i8* swiftasync %async.ctxt, %async.task* %task, %async.actor* %actor)
; CHECK-LABEL: define internal swiftcc void @polymorphic_suspend_return.resume.0(i8* {{.*}}swiftasync{{.*}} %0, i8* {{.*}}swiftself{{.*}} %1, i8* {{.*}}%2, i8* {{.*}}%3)
; CHECK: bitcast i8* %3 to %async.task*
; CHECK: }

@no_coro_suspend_fp = constant <{ i32, i32 }>
  <{ i32 trunc ( ; Relative pointer to async function
       i64 sub (
         i64 ptrtoint (void (i8*)* @no_coro_suspend to i64),
         i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32 }>, <{ i32, i32 }>* @no_coro_suspend_fp, i32 0, i32 1) to i64)
       )
     to i32),
     i32 128    ; Initial async context size without space for frame
}>

define swiftcc void @no_coro_suspend(i8* %async.ctx) "coroutine.presplit"="1" {
entry:
  %some_alloca = alloca i64
  %id = call token @llvm.coro.id.async(i32 128, i32 16, i32 0,
          i8* bitcast (<{i32, i32}>* @no_coro_suspend_fp to i8*))
  %hdl = call i8* @llvm.coro.begin(token %id, i8* null)
  call void @some_may_write(i64* %some_alloca)
  call i1 (i8*, i1, ...) @llvm.coro.end.async(i8* %hdl, i1 0)
  unreachable
}

; CHECK-LABEL: define swiftcc void @no_coro_suspend
; CHECK:   [[ALLOCA:%.*]] = alloca i64
; CHECK:   call void @some_may_write(i64* {{.*}}[[ALLOCA]])

@no_coro_suspend_swifterror_fp = constant <{ i32, i32 }>
  <{ i32 trunc ( ; Relative pointer to async function
       i64 sub (
         i64 ptrtoint (void (i8*)* @no_coro_suspend_swifterror to i64),
         i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32 }>, <{ i32, i32 }>* @no_coro_suspend_swifterror_fp, i32 0, i32 1) to i64)
       )
     to i32),
     i32 128    ; Initial async context size without space for frame
}>

declare void @do_with_swifterror(i64** swifterror)

define swiftcc void @no_coro_suspend_swifterror(i8* %async.ctx) "coroutine.presplit"="1" {
entry:
  %some_alloca = alloca swifterror i64*
  %id = call token @llvm.coro.id.async(i32 128, i32 16, i32 0,
          i8* bitcast (<{i32, i32}>* @no_coro_suspend_swifterror_fp to i8*))
  %hdl = call i8* @llvm.coro.begin(token %id, i8* null)
  store i64* null, i64** %some_alloca, align 8
  call void @do_with_swifterror(i64** swifterror %some_alloca)
  call i1 (i8*, i1, ...) @llvm.coro.end.async(i8* %hdl, i1 0)
  unreachable
}

 ; CHECK-LABEL: define swiftcc void @no_coro_suspend_swifterror
 ; CHECK:  [[ALLOCA:%.*]] = alloca swifterror i64*
 ; CHECK:   store i64* null, i64** [[ALLOCA]]
 ; CHECK:   call void @do_with_swifterror(i64** {{.*}}swifterror{{.*}} [[ALLOCA]])

@undefined_coro_async_resume_fp = constant <{ i32, i32 }>
  <{ i32 trunc (
       i64 sub (
         i64 ptrtoint (void (i8*)* @undefined_coro_async_resume to i64),
         i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32 }>, <{ i32, i32 }>* @undefined_coro_async_resume_fp, i32 0, i32 1) to i64)
       )
     to i32),
     i32 24
}>

declare void @crash()
declare void @use(i8*)

define swiftcc void @undefined_coro_async_resume(i8 *%async.ctx) "coroutine.presplit"="1" {
entry:
  %id = call token @llvm.coro.id.async(i32 24, i32 16, i32 0, i8* bitcast (<{i32, i32}>* @undefined_coro_async_resume_fp to i8*))
  %hdl = call i8* @llvm.coro.begin(token %id, i8* null)
  %undefined_resume_pointer = call i8* @llvm.coro.async.resume()
  call void @use(i8* %undefined_resume_pointer)
  call void @crash()
  %unused = call i1 (i8*, i1, ...) @llvm.coro.end.async(i8* %hdl, i1 false)
  unreachable
}
; CHECK-LABEL: define swiftcc void @undefined_coro_async_resume
; CHECK-NOT: @llvm.coro.async.resume
; CHECK: call void @use(i8* null)
; CHECK: ret

declare { i8*, i8*, i8*, i8* } @llvm.coro.suspend.async.sl_p0i8p0i8p0i8p0i8s(i32, i8*, i8*, ...)
declare i8* @llvm.coro.prepare.async(i8*)
declare token @llvm.coro.id.async(i32, i32, i32, i8*)
declare i8* @llvm.coro.begin(token, i8*)
declare i1 @llvm.coro.end.async(i8*, i1, ...)
declare i1 @llvm.coro.end(i8*, i1)
declare {i8*, i8*, i8*} @llvm.coro.suspend.async(i32, i8*, i8*, ...)
declare i8* @llvm.coro.async.context.alloc(i8*, i8*)
declare void @llvm.coro.async.context.dealloc(i8*)
declare swiftcc void @asyncReturn(i8*, %async.task*, %async.actor*)
declare swiftcc void @asyncSuspend(i8*, %async.task*, %async.actor*)
declare i8* @llvm.coro.async.resume()
declare void @llvm.coro.async.size.replace(i8*, i8*)
declare i8* @hide(i8*)

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
