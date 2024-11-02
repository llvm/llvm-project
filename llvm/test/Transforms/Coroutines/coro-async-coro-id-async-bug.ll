; RUN: opt < %s -passes='default<O2>' -S | FileCheck --check-prefixes=CHECK %s
target datalayout = "p:64:64:64"

%swift.async_func_pointer = type <{ i32, i32 }>
%swift.context = type { %swift.context*, void (%swift.context*)* }

@repoTU = global %swift.async_func_pointer <{ i32 trunc (i64 sub (i64 ptrtoint (void (%swift.context*)* @repo to i64), i64 ptrtoint (%swift.async_func_pointer* @repoTU to i64)) to i32), i32 16 }>, align 8

define swifttailcc void @repo(%swift.context* swiftasync %0) {
entry:
  %1 = alloca %swift.context*, align 8
  %2 = bitcast %swift.context* %0 to <{ %swift.context*, void (%swift.context*)* }>*
  %3 = call token @llvm.coro.id.async(i32 16, i32 16, i32 0, i8* bitcast (%swift.async_func_pointer* @repoTU to i8*))
  %4 = call i8* @llvm.coro.begin(token %3, i8* null)
  store %swift.context* %0, %swift.context** %1, align 8
  %5 = load %swift.context*, %swift.context** %1, align 8
  %6 = bitcast %swift.context* %5 to <{ %swift.context*, void (%swift.context*)* }>*
  %7 = getelementptr inbounds <{ %swift.context*, void (%swift.context*)* }>, <{ %swift.context*, void (%swift.context*)* }>* %6, i32 0, i32 1
  %8 = load void (%swift.context*)*, void (%swift.context*)** %7, align 8
  %9 = load %swift.context*, %swift.context** %1, align 8
  %10 = bitcast void (%swift.context*)* %8 to i8*
  %11 = call i1 (i8*, i1, ...) @llvm.coro.end.async(i8* %4, i1 false, void (i8*, %swift.context*)* @repo.0, i8* %10, %swift.context* %9)
  unreachable
}

; CHECK-NOT: llvm.coro.id.async

define internal swifttailcc void @repo.0(i8* %0, %swift.context* %1) #1 {
entry:
  %2 = bitcast i8* %0 to void (%swift.context*)*
  musttail call swifttailcc void %2(%swift.context* swiftasync %1)
  ret void
}

; Function Attrs: nounwind
declare token @llvm.coro.id.async(i32, i32, i32, i8*) #1

; Function Attrs: nounwind
declare i8* @llvm.coro.begin(token, i8* writeonly) #1

; Function Attrs: nounwind
declare i1 @llvm.coro.end.async(i8*, i1, ...) #1

attributes #1 = { nounwind }
