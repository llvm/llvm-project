; RUN: opt -S -passes=simplifycfg < %s | FileCheck %s --check-prefixes=CHECK

;; This used to crash:
; RUN: opt -S -passes='cgscc(simplifycfg,coro-split)' < %s
; RUN: opt < %s -O2 -S

target datalayout = "p:64:64:64"
%swift.async_func_pointer = type <{ i32, i32 }>
%swift.context = type { ptr, ptr }

declare void @f()
declare void @g()
declare void @h()
declare ptr @llvm.coro.async.resume()
declare { ptr } @llvm.coro.suspend.async.sl_p0i8s(i32, ptr, ptr, ...)
declare ptr @llvm.coro.begin(token, ptr writeonly)
declare token @llvm.coro.id.async(i32, i32, i32, ptr)

declare void @llvm.coro.end.async(ptr, i1, ...)

define linkonce_odr hidden ptr @__swift_async_resume_get_context(ptr %0) {
entry:
  ret ptr %0
}

@repoTU = hidden global %swift.async_func_pointer <{ i32 trunc (i64 sub (i64 ptrtoint (ptr @repo to i64), i64 ptrtoint (ptr @repoTU to i64)) to i32), i32 16 }>, align 8

;; Must not duplicate the suspend intrinsic because it needs to have a
;; one-to-one correspondance with the async.resume intrinsic.

; CHECK: define {{.*}}@repo
; CHECK: call {{.*}}coro.async.resume
; CHECK: call {{.*}}coro.suspend.async
; CHECK-NOT: call {{.*}}coro.suspend.async

define hidden swifttailcc void @repo(ptr swiftasync %0, i1 %cond) presplitcoroutine {
entry:
  %tok = call token @llvm.coro.id.async(i32 16, i32 16, i32 0, ptr @repoTU)
  %id = call ptr @llvm.coro.begin(token %tok, ptr null)
  call void @f()

  ;; The async.resume intrinsic models the resume function value -- the
  ;; resumption function of the split coroutine.
  %ptr0 = call ptr @llvm.coro.async.resume()

  call void @f()
  br i1 %cond, label %bb1, label %bb2

bb1:
  call void @f()
  br label %threadblock

bb2:
  call void @g()
  br label %threadblock

;; Former bug: We must not duplicate the suspend intrinsic because it needs to
;; have a one-to-one correspondance to the async.resume intrinsic (the resume
;; function continutation value).

threadblock:
  %c = phi i1 [true, %bb1], [false, %bb2]
  %t3 = call { ptr } (i32, ptr, ptr, ...) @llvm.coro.suspend.async.sl_p0i8s(i32 0, ptr %ptr0, ptr @__swift_async_resume_get_context, ptr @repo.1, ptr %ptr0, ptr %0)
  call void @h()
  br i1 %c, label %retblock, label %retblock2

retblock:
  call void (ptr, i1, ...) @llvm.coro.end.async(ptr %id, i1 false, ptr @repo.0, ptr @return, ptr %0)
  unreachable

retblock2:
  call void (ptr, i1, ...) @llvm.coro.end.async(ptr %id, i1 false)
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
