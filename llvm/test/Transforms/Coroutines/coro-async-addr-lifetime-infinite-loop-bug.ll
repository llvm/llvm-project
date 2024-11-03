; RUN: opt < %s -O0 -S | FileCheck --check-prefixes=CHECK %s

target datalayout = "p:64:64:64"

%async.task = type { i64 }
%async.actor = type { i64 }
%async.fp = type <{ i32, i32 }>

%async.ctxt = type { ptr, ptr }

; The async callee.
@my_other_async_function_fp = external global <{ i32, i32 }>
declare void @my_other_async_function(ptr %async.ctxt)

@my_async_function_fp = constant <{ i32, i32 }>
  <{ i32 trunc ( ; Relative pointer to async function
       i64 sub (
         i64 ptrtoint (ptr @my_async_function to i64),
         i64 ptrtoint (ptr getelementptr inbounds (<{ i32, i32 }>, <{ i32, i32 }>* @my_async_function_fp, i32 0, i32 1) to i64)
       )
     to i32),
     i32 128    ; Initial async context size without space for frame
}>

define swiftcc void @my_other_async_function_fp.apply(ptr %fnPtr, ptr %async.ctxt) {
  tail call swiftcc void %fnPtr(ptr %async.ctxt)
  ret void
}

declare void @escape(ptr)
declare void @store_resume(ptr)
define ptr @resume_context_projection(ptr %ctxt) {
entry:
  %resume_ctxt = load ptr, ptr %ctxt, align 8
  ret ptr %resume_ctxt
}

; The address of alloca escapes but the analysis based on lifetimes fails to see
; that it can't localize this alloca.
; CHECK: define swiftcc void @my_async_function(ptr swiftasync %async.ctxt) {
; CHECK: entry:
; CHECK-NOT: ret
; CHECK-NOT:   [[ESCAPED_ADDR:%.*]] = alloca i64, align 8
; CHECK: ret
define swiftcc void @my_async_function(ptr swiftasync %async.ctxt) {
entry:
  %escaped_addr = alloca i64

  %id = call token @llvm.coro.id.async(i32 128, i32 16, i32 0,
          ptr bitcast (<{i32, i32}>* @my_async_function_fp to ptr))
  %hdl = call ptr @llvm.coro.begin(token %id, ptr null)
  call void @llvm.lifetime.start.p0(i64 4, ptr %escaped_addr)
  call void @escape(ptr %escaped_addr)
  br label %callblock


callblock:

  %callee_context = call ptr @context_alloc()

  %resume.func_ptr = call ptr @llvm.coro.async.resume()
  call void @store_resume(ptr %resume.func_ptr)
  %res = call {ptr, ptr, ptr} (i32, ptr, ptr, ...) @llvm.coro.suspend.async(i32 0,
                                                  ptr %resume.func_ptr,
                                                  ptr @resume_context_projection,
                                                  ptr @my_other_async_function_fp.apply,
                                                  ptr @asyncSuspend, ptr %callee_context)
  br label %callblock
}

declare { ptr, ptr, ptr, ptr } @llvm.coro.suspend.async.sl_p0i8p0i8p0i8p0i8s(i32, ptr, ptr, ...)
declare ptr @llvm.coro.prepare.async(ptr)
declare token @llvm.coro.id.async(i32, i32, i32, ptr)
declare ptr @llvm.coro.begin(token, ptr)
declare i1 @llvm.coro.end.async(ptr, i1, ...)
declare i1 @llvm.coro.end(ptr, i1)
declare {ptr, ptr, ptr} @llvm.coro.suspend.async(i32, ptr, ptr, ...)
declare ptr @context_alloc()
declare void @llvm.coro.async.context.dealloc(ptr)
declare swiftcc void @asyncSuspend(ptr)
declare ptr @llvm.coro.async.resume()
declare void @llvm.coro.async.size.replace(ptr, ptr)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #0
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #0
attributes #0 = { argmemonly nofree nosync nounwind willreturn }
