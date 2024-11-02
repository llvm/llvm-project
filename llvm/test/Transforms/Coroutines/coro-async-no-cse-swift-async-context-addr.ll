; RUN: opt < %s -passes='default<O2>' -S | FileCheck --check-prefixes=CHECK %s
target datalayout = "p:64:64:64"

%swift.async_func_pointer = type <{ i32, i32 }>
%swift.context = type { %swift.context*, void (%swift.context*)* }

@repoTU = global %swift.async_func_pointer <{ i32 trunc (i64 sub (i64 ptrtoint (void (%swift.context*)* @repo to i64), i64 ptrtoint (%swift.async_func_pointer* @repoTU to i64)) to i32), i32 16 }>, align 8

declare swifttailcc void @callee.0(%swift.context* swiftasync, i8*, i64, i64)

define internal swifttailcc void @callee(i8* %0, i64 %1, i64 %2, %swift.context* %3) {
entry:
  musttail call swifttailcc void @callee.0(%swift.context* swiftasync %3, i8* %0, i64 %1, i64 %2)
  ret void
}

define swifttailcc void @repo(%swift.context* swiftasync %0) {
entry:
  %1 = alloca %swift.context*, align 8
  %2 = bitcast %swift.context* %0 to <{ %swift.context*, void (%swift.context*)* }>*
  %3 = call token @llvm.coro.id.async(i32 16, i32 16, i32 0, i8* bitcast (%swift.async_func_pointer* @repoTU to i8*))
  %4 = call i8* @llvm.coro.begin(token %3, i8* null)
  store %swift.context* %0, %swift.context** %1, align 8

  ; This context.addr is the address in the frame of the first partial function after splitting.
  %5 = call i8** @llvm.swift.async.context.addr()
	store i8* null, i8** %5, align 8

  %6 = call i8* @llvm.coro.async.resume()
  %7 = call { i8* } (i32, i8*, i8*, ...) @llvm.coro.suspend.async.sl_p0i8s(i32 0,
                                                                           i8* %6,
                                                                           i8* bitcast (i8* (i8*)* @__swift_async_resume_get_context to i8*),
                                                                           i8* bitcast (void (i8*, i64, i64, %swift.context*)* @callee to i8*),
                                                                           i8* %6, i64 0, i64 0, %swift.context* %0)
  %8 = load %swift.context*, %swift.context** %1, align 8
  %9 = bitcast %swift.context* %8 to <{ %swift.context*, void (%swift.context*)* }>*
  %10 = getelementptr inbounds <{ %swift.context*, void (%swift.context*)* }>, <{ %swift.context*, void (%swift.context*)* }>* %9, i32 0, i32 1
  %11 = load void (%swift.context*)*, void (%swift.context*)** %10, align 8
  %12 = load %swift.context*, %swift.context** %1, align 8
  %13 = bitcast void (%swift.context*)* %11 to i8*

  ; This context.addr is the address in the frame of the second partial function after splitting.
  ; It is not valid to CSE it with the previous call.
  %14 = call i8** @llvm.swift.async.context.addr()
	store i8* %13, i8** %14, align 8

  %15 = call i1 (i8*, i1, ...) @llvm.coro.end.async(i8* %4, i1 false, void (i8*, %swift.context*)* @repo.0, i8* %13, %swift.context* %12)
  unreachable
}

; Make sure we don't CSE the llvm.swift.async.context.addr calls
; CHECK: define swifttailcc void @repo
; CHECK: call i8** @llvm.swift.async.context.addr()

; CHECK: define {{.*}}swifttailcc void @repoTY0_
; CHECK: call i8** @llvm.swift.async.context.addr()

define internal swifttailcc void @repo.0(i8* %0, %swift.context* %1) #1 {
entry:
  %2 = bitcast i8* %0 to void (%swift.context*)*
  musttail call swifttailcc void %2(%swift.context* swiftasync %1)
  ret void
}

define linkonce_odr hidden i8* @__swift_async_resume_get_context(i8* %0) #1 {
entry:
  ret i8* %0
}

declare { i8* } @llvm.coro.suspend.async.sl_p0i8s(i32, i8*, i8*, ...) #1
declare token @llvm.coro.id.async(i32, i32, i32, i8*) #1
declare i8* @llvm.coro.begin(token, i8* writeonly) #1
declare i1 @llvm.coro.end.async(i8*, i1, ...) #1
declare i8* @llvm.coro.async.resume() #1
declare i8** @llvm.swift.async.context.addr() #1

attributes #1 = { nounwind }
