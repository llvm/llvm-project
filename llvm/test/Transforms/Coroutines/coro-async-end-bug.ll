; RUN: opt < %s -passes='default<O2>' -S | FileCheck --check-prefixes=CHECK %s
target datalayout = "p:64:64:64"

%swift.async_func_pointer = type <{ i32, i32 }>
%swift.context = type { ptr, ptr }
%swift.opaque = type opaque
%swift.refcounted = type { ptr, i64 }
%swift.type = type { i64 }
%swift.error = type opaque

@repoTu = linkonce_odr hidden global %swift.async_func_pointer <{ i32 trunc (i64 sub (i64 ptrtoint (ptr @repo to i64), i64 ptrtoint (ptr @repoTu to i64)) to i32), i32 16 }>, align 8

declare token @llvm.coro.id.async(i32, i32, i32, ptr) #0

declare ptr @llvm.coro.begin(token, ptr writeonly) #0

declare void @llvm.coro.end.async(ptr, i1, ...) #0

define swifttailcc void @repo(ptr swiftasync %0, ptr noalias nocapture %1, ptr noalias nocapture %2, ptr %3, ptr %4, ptr %Self, ptr %Self.AsyncSequence, ptr %Self.Element.Comparable) #1 {
entry:
  %5 = alloca ptr, align 8
  %6 = call token @llvm.coro.id.async(i32 16, i32 16, i32 0, ptr @repoTu)
  %7 = call ptr @llvm.coro.begin(token %6, ptr null)
  store ptr %0, ptr %5, align 8
  %8 = call swiftcc i1 %3(ptr noalias nocapture %1, ptr noalias nocapture %2, ptr swiftself %4) #2
  %9 = load ptr, ptr %5, align 8
  %10 = getelementptr inbounds <{ ptr, ptr }>, ptr %9, i32 0, i32 1
  %11 = load ptr, ptr %10, align 8
  %12 = load ptr, ptr %5, align 8
  call void (ptr, i1, ...) @llvm.coro.end.async(ptr %7, i1 false, ptr @repo.0, ptr %11, ptr %12, i1 %8, ptr null)
  unreachable
}

; CHECK: define swifttailcc void @repo
; CHECK-NOT:llvm.coro.end.async
; CHECK: musttail call swifttailcc void
; CHECK-NOT:llvm.coro.end.async
; CHECK-NOT: unreachable
; CHECK: ret

define internal swifttailcc void @repo.0(ptr %0, ptr %1, i1 %2, ptr %3) #0 {
entry:
  musttail call swifttailcc void %0(ptr swiftasync %1, i1 %2, ptr swiftself %3)
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind }
