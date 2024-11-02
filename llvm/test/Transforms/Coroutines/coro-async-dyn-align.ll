; RUN: opt < %s  -O0 -S | FileCheck  %s

target datalayout = "p:64:64:64"

%async.task = type { i64 }
%async.actor = type { i64 }
%async.fp = type <{ i32, i32 }>

%async.ctxt = type { i8*, void (i8*)* }

@my_other_async_function_fp = external global <{ i32, i32 }>
declare void @my_other_async_function(i8* %async.ctxt)

@my_async_function_fp = constant <{ i32, i32 }>
  <{ i32 trunc (
       i64 sub (
         i64 ptrtoint (void (i8*)* @my_async_function to i64),
         i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32 }>, <{ i32, i32 }>* @my_async_function_fp, i32 0, i32 1) to i64)
       )
     to i32),
     i32 32
}>

declare void @opaque(i64*)
declare i8* @llvm.coro.async.context.alloc(i8*, i8*)
declare void @llvm.coro.async.context.dealloc(i8*)
declare i8* @llvm.coro.async.resume()
declare token @llvm.coro.id.async(i32, i32, i32, i8*)
declare i8* @llvm.coro.begin(token, i8*)
declare i1 @llvm.coro.end.async(i8*, i1, ...)
declare i1 @llvm.coro.end(i8*, i1)
declare swiftcc void @asyncReturn(i8*)
declare swiftcc void @asyncSuspend(i8*)
declare {i8*} @llvm.coro.suspend.async(i32, i8*, i8*, ...)

define swiftcc void @my_async_function.my_other_async_function_fp.apply(i8* %fnPtr, i8* %async.ctxt) {
  %callee = bitcast i8* %fnPtr to void(i8*)*
  tail call swiftcc void %callee(i8* %async.ctxt)
  ret void
}

define i8* @__swift_async_resume_project_context(i8* %ctxt) {
entry:
  %resume_ctxt_addr = bitcast i8* %ctxt to i8**
  %resume_ctxt = load i8*, i8** %resume_ctxt_addr, align 8
  ret i8* %resume_ctxt
}


; CHECK: %my_async_function.Frame = type { i64, [48 x i8], i64, i64, [16 x i8], i8*, i64, i8* }
; CHECK: define swiftcc void @my_async_function
; CHECK:  [[T0:%.*]] = getelementptr inbounds %my_async_function.Frame, %my_async_function.Frame* %FramePtr, i32 0, i32 3
; CHECK:  [[T1:%.*]] = ptrtoint i64* [[T0]] to i64
; CHECK:  [[T2:%.*]] = add i64 [[T1]], 31
; CHECK:  [[T3:%.*]] = and i64 [[T2]], -32
; CHECK:  [[T4:%.*]] = inttoptr i64 [[T3]] to i64*
; CHECK:  [[T5:%.*]] = getelementptr inbounds %my_async_function.Frame, %my_async_function.Frame* %FramePtr, i32 0, i32 0
; CHECK:  [[T6:%.*]] = ptrtoint i64* [[T5]] to i64
; CHECK:  [[T7:%.*]] = add i64 [[T6]], 63
; CHECK:  [[T8:%.*]] = and i64 [[T7]], -64
; CHECK:  [[T9:%.*]] = inttoptr i64 [[T8]] to i64*
; CHECK:  store i64 2, i64* [[T4]]
; CHECK:  store i64 3, i64* [[T9]]

define swiftcc void @my_async_function(i8* swiftasync %async.ctxt) presplitcoroutine {
entry:
  %tmp = alloca i64, align 8
  %tmp2 = alloca i64, align 16
  %tmp3 = alloca i64, align 32
  %tmp4 = alloca i64, align 64

  %id = call token @llvm.coro.id.async(i32 32, i32 16, i32 0,
          i8* bitcast (<{i32, i32}>* @my_async_function_fp to i8*))
  %hdl = call i8* @llvm.coro.begin(token %id, i8* null)
  store i64 0, i64* %tmp
  store i64 1, i64* %tmp2
  store i64 2, i64* %tmp3
  store i64 3, i64* %tmp4

  %callee_context = call i8* @llvm.coro.async.context.alloc(i8* null, i8* null)
	%callee_context.0 = bitcast i8* %callee_context to %async.ctxt*
  %callee_context.return_to_caller.addr = getelementptr inbounds %async.ctxt, %async.ctxt* %callee_context.0, i32 0, i32 1
  %return_to_caller.addr = bitcast void(i8*)** %callee_context.return_to_caller.addr to i8**
  %resume.func_ptr = call i8* @llvm.coro.async.resume()
  store i8* %resume.func_ptr, i8** %return_to_caller.addr

  %callee = bitcast void(i8*)* @asyncSuspend to i8*
  %resume_proj_fun = bitcast i8*(i8*)* @__swift_async_resume_project_context to i8*
  %res = call {i8*} (i32, i8*, i8*, ...) @llvm.coro.suspend.async(i32 0,
                                                  i8* %resume.func_ptr,
                                                  i8* %resume_proj_fun,
                                                  void (i8*, i8*)* @my_async_function.my_other_async_function_fp.apply,
                                                  i8* %callee, i8* %callee_context)
  call void @opaque(i64* %tmp)
  call void @opaque(i64* %tmp2)
  call void @opaque(i64* %tmp3)
  call void @opaque(i64* %tmp4)
  call void @llvm.coro.async.context.dealloc(i8* %callee_context)
  tail call swiftcc void @asyncReturn(i8* %async.ctxt)
  call i1 (i8*, i1, ...) @llvm.coro.end.async(i8* %hdl, i1 0)
  unreachable
}
