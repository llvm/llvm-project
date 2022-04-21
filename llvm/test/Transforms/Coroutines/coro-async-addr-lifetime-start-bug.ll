; RUN: opt < %s -O0 -S | FileCheck --check-prefixes=CHECK %s

target datalayout = "p:64:64:64"

%async.task = type { i64 }
%async.actor = type { i64 }
%async.fp = type <{ i32, i32 }>

%async.ctxt = type { i8*, void (i8*, %async.task*, %async.actor*)* }

; The async callee.
@my_other_async_function_fp = external global <{ i32, i32 }>
declare void @my_other_async_function(i8* %async.ctxt)

@my_async_function_fp = constant <{ i32, i32 }>
  <{ i32 trunc ( ; Relative pointer to async function
       i64 sub (
         i64 ptrtoint (void (i8*)* @my_async_function to i64),
         i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32 }>, <{ i32, i32 }>* @my_async_function_fp, i32 0, i32 1) to i64)
       )
     to i32),
     i32 128    ; Initial async context size without space for frame
}>

define swiftcc void @my_other_async_function_fp.apply(i8* %fnPtr, i8* %async.ctxt) {
  %callee = bitcast i8* %fnPtr to void(i8*)*
  tail call swiftcc void %callee(i8* %async.ctxt)
  ret void
}

declare void @escape(i64*)
declare void @store_resume(i8*)
declare i1 @exitLoop()
define i8* @resume_context_projection(i8* %ctxt) {
entry:
  %resume_ctxt_addr = bitcast i8* %ctxt to i8**
  %resume_ctxt = load i8*, i8** %resume_ctxt_addr, align 8
  ret i8* %resume_ctxt
}

define swiftcc void @my_async_function(i8* swiftasync %async.ctxt) {
entry:
  %escaped_addr = alloca i64

  %id = call token @llvm.coro.id.async(i32 128, i32 16, i32 0,
          i8* bitcast (<{i32, i32}>* @my_async_function_fp to i8*))
  %hdl = call i8* @llvm.coro.begin(token %id, i8* null)
  %ltb = bitcast i64* %escaped_addr to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %ltb)
  br label %callblock


callblock:

  %callee_context = call i8* @context_alloc()

  %resume.func_ptr = call i8* @llvm.coro.async.resume()
  call void @store_resume(i8* %resume.func_ptr)
  %resume_proj_fun = bitcast i8*(i8*)* @resume_context_projection to i8*
  %callee = bitcast void(i8*)* @asyncSuspend to i8*
  %res = call {i8*, i8*, i8*} (i32, i8*, i8*, ...) @llvm.coro.suspend.async(i32 0,
                                                  i8* %resume.func_ptr,
                                                  i8* %resume_proj_fun,
                                                  void (i8*, i8*)* @my_other_async_function_fp.apply,
                                                  i8* %callee, i8* %callee_context)
  call void @escape(i64* %escaped_addr)
  %exitCond = call i1 @exitLoop()

;; We used to move the lifetime.start intrinsic here =>
;; This exposes two bugs:
;  1.) The code should use the basic block start not end as insertion point
;  More problematically:
;  2.) The code marks the stack object as not alive for part of the loop.

  br i1 %exitCond, label %loop_exit, label %loop
  %res2 = call {i8*, i8*, i8*} (i32, i8*, i8*, ...) @llvm.coro.suspend.async(i32 0,
                                                  i8* %resume.func_ptr,
                                                  i8* %resume_proj_fun,
                                                  void (i8*, i8*)* @my_other_async_function_fp.apply,
                                                  i8* %callee, i8* %callee_context)
 
  %exitCond2 = call i1 @exitLoop()
  br i1 %exitCond2, label %loop_exit, label %loop

loop:
  br label %callblock

loop_exit:
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %ltb)
  call i1 (i8*, i1, ...) @llvm.coro.end.async(i8* %hdl, i1 false)
  unreachable
}

; CHECK: define {{.*}} void @my_async_function.resume.0(
; CHECK-NOT:  call void @llvm.lifetime.start.p0i8(i64 4, i8* %3)
; CHECK:  br i1 %exitCond, label %loop_exit, label %loop
; CHECK: lifetime.end
; CHECK: }

declare { i8*, i8*, i8*, i8* } @llvm.coro.suspend.async.sl_p0i8p0i8p0i8p0i8s(i32, i8*, i8*, ...)
declare i8* @llvm.coro.prepare.async(i8*)
declare token @llvm.coro.id.async(i32, i32, i32, i8*)
declare i8* @llvm.coro.begin(token, i8*)
declare i1 @llvm.coro.end.async(i8*, i1, ...)
declare i1 @llvm.coro.end(i8*, i1)
declare {i8*, i8*, i8*} @llvm.coro.suspend.async(i32, i8*, i8*, ...)
declare i8* @context_alloc()
declare void @llvm.coro.async.context.dealloc(i8*)
declare swiftcc void @asyncSuspend(i8*)
declare i8* @llvm.coro.async.resume()
declare void @llvm.coro.async.size.replace(i8*, i8*)
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #0
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #0
attributes #0 = { argmemonly nofree nosync nounwind willreturn }
