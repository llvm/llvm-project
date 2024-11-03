; RUN: opt < %s -O2 -S
; RUN: opt -S -hoist-common-insts -hoist-common-insts -passes=simplifycfg < %s | FileCheck %s --check-prefixes=CHECK
target datalayout = "p:64:64:64"
%swift.async_func_pointer = type <{ i32, i32 }>
%swift.context = type { ptr, ptr }

declare void @f()
declare void @g()
declare ptr @llvm.coro.async.resume()
declare { ptr } @llvm.coro.suspend.async.sl_p0i8s(i32, ptr, ptr, ...)
declare ptr @llvm.coro.begin(token, ptr writeonly)
declare token @llvm.coro.id.async(i32, i32, i32, ptr)

declare i1 @llvm.coro.end.async(ptr, i1, ...)

define linkonce_odr hidden ptr @__swift_async_resume_get_context(ptr %0) {
entry:
  ret ptr %0
}

@repoTU = hidden global %swift.async_func_pointer <{ i32 trunc (i64 sub (i64 ptrtoint (ptr @repo to i64), i64 ptrtoint (ptr @repoTU to i64)) to i32), i32 16 }>, align 8

; This test used to crash in optimized mode because simplify cfg would sink the
; suspend.async() instruction into a common successor block.

; CHECK: swifttailcc void @repo
; CHECK:llvm.coro.suspend.async.sl_p0s
; CHECK: br
; CHECK:llvm.coro.suspend.async.sl_p0s
; CHECK: br
; CHECK: ret

define hidden swifttailcc void @repo(ptr swiftasync %0, i1 %cond) {
entry:
  %tok = call token @llvm.coro.id.async(i32 16, i32 16, i32 0, ptr @repoTU)
  %id = call ptr @llvm.coro.begin(token %tok, ptr null)
  br i1 %cond, label %bb1, label %bb2

bb1:
  call void @f()
  %ptr0 = call ptr @llvm.coro.async.resume()
  call void @f()
  ; Simplifycfg must not sink the suspend instruction.
  %t3 = call { ptr } (i32, ptr, ptr, ...) @llvm.coro.suspend.async.sl_p0i8s(i32 0, ptr %ptr0, ptr @__swift_async_resume_get_context, ptr @repo.1, ptr %ptr0, ptr %0)
  br label %tailblock

bb2:
  call void @g()
  %ptr1 = call ptr @llvm.coro.async.resume()
  call void @g()
  ; Simplifycfg must not sink the suspend instruction.
  %t4 = call { ptr } (i32, ptr, ptr, ...) @llvm.coro.suspend.async.sl_p0i8s(i32 0, ptr %ptr1, ptr @__swift_async_resume_get_context, ptr @repo.1, ptr %ptr1, ptr %0)
  br label %tailblock

tailblock:
  %t = call i1 (ptr, i1, ...) @llvm.coro.end.async(ptr %id, i1 false, ptr @repo.0, ptr @return, ptr %0)
  unreachable
}

define internal swifttailcc void @repo.0(ptr %0, ptr %1) {
entry:
  musttail call swifttailcc void %0(ptr swiftasync %1)
  ret void
}

declare swifttailcc void @swift_task_switch(ptr, ptr)

define internal swifttailcc void @repo.1(ptr %0, ptr %1) {
entry:
  musttail call swifttailcc void @swift_task_switch(ptr swiftasync %1, ptr %0)
  ret void
}

declare swifttailcc void @return(ptr swiftasync)

@repoTU2 = hidden global %swift.async_func_pointer <{ i32 trunc (i64 sub (i64 ptrtoint (ptr @repo2 to i64), i64 ptrtoint (ptr @repoTU2 to i64)) to i32), i32 16 }>, align 8

; This test used to crash in optimized mode because simplify cfg would hoist the
; async.resume() instruction into a common block.

; CHECK: swifttailcc void @repo2
; CHECK: entry:
; CHECK: br i1

; CHECK: @llvm.coro.async.resume()
; CHECK: llvm.coro.suspend.async.sl_p0s
; CHECK: br

; CHECK:@llvm.coro.async.resume()
; CHECK:llvm.coro.suspend.async.sl_p0s
; CHECK: br

; CHECK: ret

define hidden swifttailcc void @repo2(ptr swiftasync %0, i1 %cond) {
entry:
  %tok = call token @llvm.coro.id.async(i32 16, i32 16, i32 0, ptr @repoTU2)
  %id = call ptr @llvm.coro.begin(token %tok, ptr null)
  br i1 %cond, label %bb1, label %bb2

bb1:
  ; Simplifycfg must not hoist the resume instruction.
  %ptr0 = call ptr @llvm.coro.async.resume()
  call void @f()
  %t3 = call { ptr } (i32, ptr, ptr, ...) @llvm.coro.suspend.async.sl_p0i8s(i32 0, ptr %ptr0, ptr @__swift_async_resume_get_context, ptr @repo.1, ptr %ptr0, ptr %0)
  call void @f()
  br label %tailblock

bb2:
  ; Simplifycfg must not hoist the resume instruction.
  %ptr1 = call ptr @llvm.coro.async.resume()
  call void @g()
  %t4 = call { ptr } (i32, ptr, ptr, ...) @llvm.coro.suspend.async.sl_p0i8s(i32 0, ptr %ptr1, ptr @__swift_async_resume_get_context, ptr @repo.1, ptr %ptr1, ptr %0)
  call void @g()
  br label %tailblock

tailblock:
  %t = call i1 (ptr, i1, ...) @llvm.coro.end.async(ptr %id, i1 false, ptr @repo.0, ptr @return, ptr %0)
  unreachable
}
