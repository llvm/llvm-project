; RUN: opt < %s -passes='default<O2>' -S | FileCheck --check-prefixes=CHECK %s
target datalayout = "p:64:64:64"

%swift.async_func_pointer = type <{ i32, i32 }>
%swift.context = type { ptr, ptr }

@repoTU = global %swift.async_func_pointer <{ i32 trunc (i64 sub (i64 ptrtoint (ptr @repo to i64), i64 ptrtoint (ptr @repoTU to i64)) to i32), i32 16 }>, align 8

define swifttailcc void @repo(ptr swiftasync %0) {
entry:
  %1 = alloca ptr, align 8
  %2 = call token @llvm.coro.id.async(i32 16, i32 16, i32 0, ptr @repoTU)
  %3 = call ptr @llvm.coro.begin(token %2, ptr null)
  store ptr %0, ptr %1, align 8
  %4 = load ptr, ptr %1, align 8
  %5 = getelementptr inbounds <{ ptr, ptr }>, <{ ptr, ptr }>* %4, i32 0, i32 1
  %6 = load ptr, ptr %5, align 8
  %7 = load ptr, ptr %1, align 8
  %8 = call i1 (ptr, i1, ...) @llvm.coro.end.async(ptr %3, i1 false, ptr @repo.0, ptr %6, ptr %7)
  unreachable
}

; CHECK-NOT: llvm.coro.id.async

define internal swifttailcc void @repo.0(ptr %0, ptr %1) #1 {
entry:
  musttail call swifttailcc void %0(ptr swiftasync %1)
  ret void
}

; Function Attrs: nounwind
declare token @llvm.coro.id.async(i32, i32, i32, ptr) #1

; Function Attrs: nounwind
declare ptr @llvm.coro.begin(token, ptr writeonly) #1

; Function Attrs: nounwind
declare i1 @llvm.coro.end.async(ptr, i1, ...) #1

attributes #1 = { nounwind }
