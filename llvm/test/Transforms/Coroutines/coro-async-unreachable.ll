; RUN: opt < %s -passes='default<O2>' -S | FileCheck --check-prefixes=CHECK %s

target datalayout = "p:64:64:64"

%async.task = type { i64 }
%async.actor = type { i64 }
%async.fp = type <{ i32, i32 }>

%async.ctxt = type { ptr, ptr }

; The async callee.
@my_other_async_function_fp = external global <{ i32, i32 }>
declare void @my_other_async_function(ptr %async.ctxt)

; Function that implements the dispatch to the callee function.
define swiftcc void @my_async_function.my_other_async_function_fp.apply(ptr %fnPtr, ptr %async.ctxt, ptr %task, ptr %actor) {
  tail call swiftcc void %fnPtr(ptr %async.ctxt, ptr %task, ptr %actor)
  ret void
}

declare void @some_user(i64)
declare void @some_may_write(ptr)

define ptr @resume_context_projection(ptr %ctxt) {
entry:
  %resume_ctxt = load ptr, ptr %ctxt, align 8
  ret ptr %resume_ctxt
}


@unreachable_fp = constant <{ i32, i32 }>
  <{ i32 trunc ( ; Relative pointer to async function
       i64 sub (
         i64 ptrtoint (ptr @unreachable to i64),
         i64 ptrtoint (ptr getelementptr inbounds (<{ i32, i32 }>, <{ i32, i32 }>* @unreachable_fp, i32 0, i32 1) to i64)
       )
     to i32),
     i32 128    ; Initial async context size without space for frame
}>

define swiftcc void @unreachable(ptr %async.ctxt, ptr %task, ptr %actor)  {
entry:
  %tmp = alloca { i64, i64 }, align 8
  %proj.1 = getelementptr inbounds { i64, i64 }, ptr %tmp, i64 0, i32 0
  %proj.2 = getelementptr inbounds { i64, i64 }, ptr %tmp, i64 0, i32 1

  %id = call token @llvm.coro.id.async(i32 128, i32 16, i32 0,
          ptr bitcast (<{i32, i32}>* @unreachable_fp to ptr))
  %hdl = call ptr @llvm.coro.begin(token %id, ptr null)
  store i64 0, ptr %proj.1, align 8
  store i64 1, ptr %proj.2, align 8
  call void @some_may_write(ptr %proj.1)

	; Begin lowering: apply %my_other_async_function(%args...)
  ; setup callee context
  %arg1 = bitcast <{ i32, i32}>* @my_other_async_function_fp to ptr
  %callee_context = call ptr @llvm.coro.async.context.alloc(ptr %task, ptr %arg1)
  ; store the return continuation
  %callee_context.return_to_caller.addr = getelementptr inbounds %async.ctxt, ptr %callee_context, i32 0, i32 1
  %resume.func_ptr = call ptr @llvm.coro.async.resume()
  store ptr %resume.func_ptr, ptr %callee_context.return_to_caller.addr

  ; store caller context into callee context
  store ptr %async.ctxt, ptr %callee_context
  %res = call {ptr, ptr, ptr} (i32, ptr, ptr, ...) @llvm.coro.suspend.async(i32 0,
                                                  ptr %resume.func_ptr,
                                                  ptr @resume_context_projection,
                                                  ptr @my_async_function.my_other_async_function_fp.apply,
                                                  ptr @asyncSuspend, ptr %callee_context, ptr %task, ptr %actor)

  call void @llvm.coro.async.context.dealloc(ptr %callee_context)
  %continuation_task_arg = extractvalue {ptr, ptr, ptr} %res, 1
  %val = load i64, ptr %proj.1
  call void @some_user(i64 %val)
  %val.2 = load i64, ptr %proj.2
  call void @some_user(i64 %val.2)
  unreachable
}

; CHECK: define swiftcc void @unreachable
; CHECK-NOT: @llvm.coro.suspend.async
; CHECK: return

; CHECK: define internal swiftcc void @unreachable.resume.0
; CHECK: unreachable

declare ptr @llvm.coro.prepare.async(ptr)
declare token @llvm.coro.id.async(i32, i32, i32, ptr)
declare ptr @llvm.coro.begin(token, ptr)
declare {ptr, ptr, ptr} @llvm.coro.suspend.async(i32, ptr, ptr, ...)
declare ptr @llvm.coro.async.context.alloc(ptr, ptr)
declare void @llvm.coro.async.context.dealloc(ptr)
declare swiftcc void @asyncReturn(ptr, ptr, ptr)
declare swiftcc void @asyncSuspend(ptr, ptr, ptr)
declare ptr @llvm.coro.async.resume()
