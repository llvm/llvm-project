; RUN: opt < %s -passes='default<O2>' -S | FileCheck --check-prefixes=CHECK %s
target datalayout = "p:64:64:64"

%swift.async_func_pointer = type <{ i32, i32 }>
%swift.context = type { ptr, ptr }

@repoTU = global %swift.async_func_pointer <{ i32 trunc (i64 sub (i64 ptrtoint (ptr @repo to i64), i64 ptrtoint (ptr @repoTU to i64)) to i32), i32 16 }>, align 8

declare swifttailcc void @callee.0(ptr swiftasync, ptr, i64, i64)

define internal swifttailcc void @callee(ptr %0, i64 %1, i64 %2, ptr %3) {
entry:
  musttail call swifttailcc void @callee.0(ptr swiftasync %3, ptr %0, i64 %1, i64 %2)
  ret void
}

define swifttailcc void @repo(ptr swiftasync %0) {
entry:
  %1 = alloca ptr, align 8
  %2 = call token @llvm.coro.id.async(i32 16, i32 16, i32 0, ptr @repoTU)
  %3 = call ptr @llvm.coro.begin(token %2, ptr null)
  store ptr %0, ptr %1, align 8

  ; This context.addr is the address in the frame of the first partial function after splitting.
  %4 = call ptr @llvm.swift.async.context.addr()
	store ptr null, ptr %4, align 8

  %5 = call ptr @llvm.coro.async.resume()
  %6 = call { ptr } (i32, ptr, ptr, ...) @llvm.coro.suspend.async.sl_p0i8s(i32 0,
                                                                           ptr %5,
                                                                           ptr @__swift_async_resume_get_context,
                                                                           ptr @callee,
                                                                           ptr %5, i64 0, i64 0, ptr %0)
  %7 = load ptr, ptr %1, align 8
  %8 = getelementptr inbounds <{ ptr, ptr }>, ptr %7, i32 0, i32 1
  %9 = load ptr, ptr %8, align 8
  %10 = load ptr, ptr %1, align 8

  ; This context.addr is the address in the frame of the second partial function after splitting.
  ; It is not valid to CSE it with the previous call.
  %11 = call ptr @llvm.swift.async.context.addr()
	store ptr %9, ptr %11, align 8

  call void (ptr, i1, ...) @llvm.coro.end.async(ptr %3, i1 false, ptr @repo.0, ptr %9, ptr %10)
  unreachable
}

; Make sure we don't CSE the llvm.swift.async.context.addr calls
; CHECK: define swifttailcc void @repo
; CHECK: call ptr @llvm.swift.async.context.addr()

; CHECK: define {{.*}}swifttailcc void @repoTY0_
; CHECK: call ptr @llvm.swift.async.context.addr()

define internal swifttailcc void @repo.0(ptr %0, ptr %1) #1 {
entry:
  musttail call swifttailcc void %0(ptr swiftasync %1)
  ret void
}

define linkonce_odr hidden ptr @__swift_async_resume_get_context(ptr %0) #1 {
entry:
  ret ptr %0
}

declare { ptr } @llvm.coro.suspend.async.sl_p0i8s(i32, ptr, ptr, ...) #1
declare token @llvm.coro.id.async(i32, i32, i32, ptr) #1
declare ptr @llvm.coro.begin(token, ptr writeonly) #1
declare void @llvm.coro.end.async(ptr, i1, ...) #1
declare ptr @llvm.coro.async.resume() #1
declare ptr @llvm.swift.async.context.addr() #1

attributes #1 = { nounwind }
