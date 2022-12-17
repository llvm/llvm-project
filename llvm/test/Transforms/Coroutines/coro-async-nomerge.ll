; RUN: opt < %s -O2 -S
; RUN: opt -S -hoist-common-insts -hoist-common-insts -passes=simplifycfg < %s | FileCheck %s --check-prefixes=CHECK
target datalayout = "p:64:64:64"
%swift.async_func_pointer = type <{ i32, i32 }>
%swift.context = type { %swift.context*, void (%swift.context*)* }

declare void @f()
declare void @g()
declare i8* @llvm.coro.async.resume()
declare { i8* } @llvm.coro.suspend.async.sl_p0i8s(i32, i8*, i8*, ...)
declare i8* @llvm.coro.begin(token, i8* writeonly)
declare token @llvm.coro.id.async(i32, i32, i32, i8*)

declare i1 @llvm.coro.end.async(i8*, i1, ...)

define linkonce_odr hidden i8* @__swift_async_resume_get_context(i8* %0) {
entry:
  ret i8* %0
}

@repoTU = hidden global %swift.async_func_pointer <{ i32 trunc (i64 sub (i64 ptrtoint (void (%swift.context*, i1)* @repo to i64), i64 ptrtoint (%swift.async_func_pointer* @repoTU to i64)) to i32), i32 16 }>, align 8

; This test used to crash in optimized mode because simplify cfg would sink the
; suspend.async() instruction into a common successor block.

; CHECK: swifttailcc void @repo
; CHECK:llvm.coro.suspend.async.sl_p0i8s
; CHECK: br
; CHECK:llvm.coro.suspend.async.sl_p0i8s
; CHECK: br
; CHECK: ret

define hidden swifttailcc void @repo(%swift.context* swiftasync %0, i1 %cond) {
entry:
  %tok = call token @llvm.coro.id.async(i32 16, i32 16, i32 0, i8* bitcast (%swift.async_func_pointer* @repoTU to i8*))
  %id = call i8* @llvm.coro.begin(token %tok, i8* null)
  br i1 %cond, label %bb1, label %bb2

bb1:
  call void @f()
  %ptr0 = call i8* @llvm.coro.async.resume()
  call void @f()
  ; Simplifycfg must not sink the suspend instruction.
  %t3 = call { i8* } (i32, i8*, i8*, ...) @llvm.coro.suspend.async.sl_p0i8s(i32 0, i8* %ptr0, i8* bitcast (i8* (i8*)* @__swift_async_resume_get_context to i8*), i8* bitcast (void (i8*, %swift.context*)* @repo.1 to i8*), i8* %ptr0, %swift.context* %0)
  br label %tailblock

bb2:
  call void @g()
  %ptr1 = call i8* @llvm.coro.async.resume()
  call void @g()
  ; Simplifycfg must not sink the suspend instruction.
  %t4 = call { i8* } (i32, i8*, i8*, ...) @llvm.coro.suspend.async.sl_p0i8s(i32 0, i8* %ptr1, i8* bitcast (i8* (i8*)* @__swift_async_resume_get_context to i8*), i8* bitcast (void (i8*, %swift.context*)* @repo.1 to i8*), i8* %ptr1, %swift.context* %0)
  br label %tailblock

tailblock:
  %t = call i1 (i8*, i1, ...) @llvm.coro.end.async(i8* %id, i1 false, void (i8*, %swift.context*)* @repo.0, i8* bitcast (void (%swift.context*)* @return to i8*), %swift.context* %0)
  unreachable
}

define internal swifttailcc void @repo.0(i8* %0, %swift.context* %1) {
entry:
  %2 = bitcast i8* %0 to void (%swift.context*)*
  musttail call swifttailcc void %2(%swift.context* swiftasync %1)
  ret void
}

declare swifttailcc void @swift_task_switch(%swift.context*, i8*)

define internal swifttailcc void @repo.1(i8* %0, %swift.context* %1) {
entry:
  musttail call swifttailcc void @swift_task_switch(%swift.context* swiftasync %1, i8* %0)
  ret void
}

declare swifttailcc void @return(%swift.context* swiftasync)

@repoTU2 = hidden global %swift.async_func_pointer <{ i32 trunc (i64 sub (i64 ptrtoint (void (%swift.context*, i1)* @repo2 to i64), i64 ptrtoint (%swift.async_func_pointer* @repoTU2 to i64)) to i32), i32 16 }>, align 8

; This test used to crash in optimized mode because simplify cfg would hoist the
; async.resume() instruction into a common block.

; CHECK: swifttailcc void @repo2
; CHECK: entry:
; CHECK: br i1

; CHECK: @llvm.coro.async.resume()
; CHECK: llvm.coro.suspend.async.sl_p0i8s
; CHECK: br

; CHECK:@llvm.coro.async.resume()
; CHECK:llvm.coro.suspend.async.sl_p0i8s
; CHECK: br

; CHECK: ret

define hidden swifttailcc void @repo2(%swift.context* swiftasync %0, i1 %cond) {
entry:
  %tok = call token @llvm.coro.id.async(i32 16, i32 16, i32 0, i8* bitcast (%swift.async_func_pointer* @repoTU2 to i8*))
  %id = call i8* @llvm.coro.begin(token %tok, i8* null)
  br i1 %cond, label %bb1, label %bb2

bb1:
  ; Simplifycfg must not hoist the resume instruction.
  %ptr0 = call i8* @llvm.coro.async.resume()
  call void @f()
  %t3 = call { i8* } (i32, i8*, i8*, ...) @llvm.coro.suspend.async.sl_p0i8s(i32 0, i8* %ptr0, i8* bitcast (i8* (i8*)* @__swift_async_resume_get_context to i8*), i8* bitcast (void (i8*, %swift.context*)* @repo.1 to i8*), i8* %ptr0, %swift.context* %0)
  call void @f()
  br label %tailblock

bb2:
  ; Simplifycfg must not hoist the resume instruction.
  %ptr1 = call i8* @llvm.coro.async.resume()
  call void @g()
  %t4 = call { i8* } (i32, i8*, i8*, ...) @llvm.coro.suspend.async.sl_p0i8s(i32 0, i8* %ptr1, i8* bitcast (i8* (i8*)* @__swift_async_resume_get_context to i8*), i8* bitcast (void (i8*, %swift.context*)* @repo.1 to i8*), i8* %ptr1, %swift.context* %0)
  call void @g()
  br label %tailblock

tailblock:
  %t = call i1 (i8*, i1, ...) @llvm.coro.end.async(i8* %id, i1 false, void (i8*, %swift.context*)* @repo.0, i8* bitcast (void (%swift.context*)* @return to i8*), %swift.context* %0)
  unreachable
}
