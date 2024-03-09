; RUN: opt < %s  -O0 -S | FileCheck  %s

target datalayout = "p:64:64:64"

%async.task = type { i64 }
%async.actor = type { i64 }
%async.fp = type <{ i32, i32 }>

%async.ctxt = type { ptr, ptr }

@my_other_async_function_fp = external global <{ i32, i32 }>
declare void @my_other_async_function(ptr %async.ctxt)

@my_async_function_fp = constant <{ i32, i32 }>
  <{ i32 trunc (
       i64 sub (
         i64 ptrtoint (ptr @my_async_function to i64),
         i64 ptrtoint (ptr getelementptr inbounds (<{ i32, i32 }>, ptr @my_async_function_fp, i32 0, i32 1) to i64)
       )
     to i32),
     i32 32
}>

declare void @opaque(ptr)
declare ptr @llvm.coro.async.context.alloc(ptr, ptr)
declare void @llvm.coro.async.context.dealloc(ptr)
declare ptr @llvm.coro.async.resume()
declare token @llvm.coro.id.async(i32, i32, i32, ptr)
declare ptr @llvm.coro.begin(token, ptr)
declare i1 @llvm.coro.end.async(ptr, i1, ...)
declare i1 @llvm.coro.end(ptr, i1, token)
declare swiftcc void @asyncReturn(ptr)
declare swiftcc void @asyncSuspend(ptr)
declare {ptr} @llvm.coro.suspend.async(i32, ptr, ptr, ...)

define swiftcc void @my_async_function.my_other_async_function_fp.apply(ptr %fnPtr, ptr %async.ctxt) {
  tail call swiftcc void %fnPtr(ptr %async.ctxt)
  ret void
}

define ptr @__swift_async_resume_project_context(ptr %ctxt) {
entry:
  %resume_ctxt = load ptr, ptr %ctxt, align 8
  ret ptr %resume_ctxt
}


; CHECK: %my_async_function.Frame = type { i64, [48 x i8], i64, i64, [16 x i8], ptr, i64, ptr }
; CHECK: define swiftcc void @my_async_function
; CHECK:  [[T0:%.*]] = getelementptr inbounds %my_async_function.Frame, ptr %async.ctx.frameptr, i32 0, i32 3
; CHECK:  [[T1:%.*]] = ptrtoint ptr [[T0]] to i64
; CHECK:  [[T2:%.*]] = add i64 [[T1]], 31
; CHECK:  [[T3:%.*]] = and i64 [[T2]], -32
; CHECK:  [[T4:%.*]] = inttoptr i64 [[T3]] to ptr
; CHECK:  [[FP:%.*]] = getelementptr inbounds %my_async_function.Frame, ptr %async.ctx.frameptr, i32 0, i32 0
; CHECK:  [[T6:%.*]] = ptrtoint ptr [[FP]] to i64
; CHECK:  [[T7:%.*]] = add i64 [[T6]], 63
; CHECK:  [[T8:%.*]] = and i64 [[T7]], -64
; CHECK:  [[T9:%.*]] = inttoptr i64 [[T8]] to ptr
; CHECK:  store i64 2, ptr [[T4]]
; CHECK:  store i64 3, ptr [[T9]]

define swiftcc void @my_async_function(ptr swiftasync %async.ctxt) presplitcoroutine {
entry:
  %tmp = alloca i64, align 8
  %tmp2 = alloca i64, align 16
  %tmp3 = alloca i64, align 32
  %tmp4 = alloca i64, align 64

  %id = call token @llvm.coro.id.async(i32 32, i32 16, i32 0,
          ptr @my_async_function_fp)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr null)
  store i64 0, ptr %tmp
  store i64 1, ptr %tmp2
  store i64 2, ptr %tmp3
  store i64 3, ptr %tmp4

  %callee_context = call ptr @llvm.coro.async.context.alloc(ptr null, ptr null)
  %callee_context.return_to_caller.addr = getelementptr inbounds %async.ctxt, ptr %callee_context, i32 0, i32 1
  %resume.func_ptr = call ptr @llvm.coro.async.resume()
  store ptr %resume.func_ptr, ptr %callee_context.return_to_caller.addr

  %res = call {ptr} (i32, ptr, ptr, ...) @llvm.coro.suspend.async(i32 0,
                                                  ptr %resume.func_ptr,
                                                  ptr @__swift_async_resume_project_context,
                                                  ptr @my_async_function.my_other_async_function_fp.apply,
                                                  ptr @asyncSuspend, ptr %callee_context)
  call void @opaque(ptr %tmp)
  call void @opaque(ptr %tmp2)
  call void @opaque(ptr %tmp3)
  call void @opaque(ptr %tmp4)
  call void @llvm.coro.async.context.dealloc(ptr %callee_context)
  tail call swiftcc void @asyncReturn(ptr %async.ctxt)
  call i1 (ptr, i1, ...) @llvm.coro.end.async(ptr %hdl, i1 0)
  unreachable
}
