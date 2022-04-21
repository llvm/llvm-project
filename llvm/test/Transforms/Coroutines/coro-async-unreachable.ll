; RUN: opt < %s -passes='default<O2>' -S | FileCheck --check-prefixes=CHECK %s

target datalayout = "p:64:64:64"

%async.task = type { i64 }
%async.actor = type { i64 }
%async.fp = type <{ i32, i32 }>

%async.ctxt = type { i8*, void (i8*, %async.task*, %async.actor*)* }

; The async callee.
@my_other_async_function_fp = external global <{ i32, i32 }>
declare void @my_other_async_function(i8* %async.ctxt)

; Function that implements the dispatch to the callee function.
define swiftcc void @my_async_function.my_other_async_function_fp.apply(i8* %fnPtr, i8* %async.ctxt, %async.task* %task, %async.actor* %actor) {
  %callee = bitcast i8* %fnPtr to void(i8*, %async.task*, %async.actor*)*
  tail call swiftcc void %callee(i8* %async.ctxt, %async.task* %task, %async.actor* %actor)
  ret void
}

declare void @some_user(i64)
declare void @some_may_write(i64*)

define i8* @resume_context_projection(i8* %ctxt) {
entry:
  %resume_ctxt_addr = bitcast i8* %ctxt to i8**
  %resume_ctxt = load i8*, i8** %resume_ctxt_addr, align 8
  ret i8* %resume_ctxt
}


@unreachable_fp = constant <{ i32, i32 }>
  <{ i32 trunc ( ; Relative pointer to async function
       i64 sub (
         i64 ptrtoint (void (i8*, %async.task*, %async.actor*)* @unreachable to i64),
         i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32 }>, <{ i32, i32 }>* @unreachable_fp, i32 0, i32 1) to i64)
       )
     to i32),
     i32 128    ; Initial async context size without space for frame
}>

define swiftcc void @unreachable(i8* %async.ctxt, %async.task* %task, %async.actor* %actor)  {
entry:
  %tmp = alloca { i64, i64 }, align 8
  %proj.1 = getelementptr inbounds { i64, i64 }, { i64, i64 }* %tmp, i64 0, i32 0
  %proj.2 = getelementptr inbounds { i64, i64 }, { i64, i64 }* %tmp, i64 0, i32 1

  %id = call token @llvm.coro.id.async(i32 128, i32 16, i32 0,
          i8* bitcast (<{i32, i32}>* @unreachable_fp to i8*))
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
  %res = call {i8*, i8*, i8*} (i32, i8*, i8*, ...) @llvm.coro.suspend.async(i32 0,
                                                  i8* %resume.func_ptr,
                                                  i8* %resume_proj_fun,
                                                  void (i8*, i8*, %async.task*, %async.actor*)* @my_async_function.my_other_async_function_fp.apply,
                                                  i8* %callee, i8* %callee_context, %async.task* %task, %async.actor *%actor)

  call void @llvm.coro.async.context.dealloc(i8* %callee_context)
  %continuation_task_arg = extractvalue {i8*, i8*, i8*} %res, 1
  %task.2 =  bitcast i8* %continuation_task_arg to %async.task*
  %val = load i64, i64* %proj.1
  call void @some_user(i64 %val)
  %val.2 = load i64, i64* %proj.2
  call void @some_user(i64 %val.2)
  unreachable
}

; CHECK: define swiftcc void @unreachable
; CHECK-NOT: @llvm.coro.suspend.async
; CHECK: return

; CHECK: define internal swiftcc void @unreachable.resume.0
; CHECK: unreachable

declare i8* @llvm.coro.prepare.async(i8*)
declare token @llvm.coro.id.async(i32, i32, i32, i8*)
declare i8* @llvm.coro.begin(token, i8*)
declare {i8*, i8*, i8*} @llvm.coro.suspend.async(i32, i8*, i8*, ...)
declare i8* @llvm.coro.async.context.alloc(i8*, i8*)
declare void @llvm.coro.async.context.dealloc(i8*)
declare swiftcc void @asyncReturn(i8*, %async.task*, %async.actor*)
declare swiftcc void @asyncSuspend(i8*, %async.task*, %async.actor*)
declare i8* @llvm.coro.async.resume()
