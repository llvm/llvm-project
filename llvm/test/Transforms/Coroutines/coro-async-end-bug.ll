; RUN: opt < %s -passes='default<O2>' -S | FileCheck --check-prefixes=CHECK %s
target datalayout = "p:64:64:64"

%swift.async_func_pointer = type <{ i32, i32 }>
%swift.context = type { %swift.context*, void (%swift.context*)* }
%swift.opaque = type opaque
%swift.refcounted = type { %swift.type*, i64 }
%swift.type = type { i64 }
%swift.error = type opaque

@repoTu = linkonce_odr hidden global %swift.async_func_pointer <{ i32 trunc (i64 sub (i64 ptrtoint (void (%swift.context*, %swift.opaque*, %swift.opaque*, i8*, %swift.refcounted*, %swift.type*, i8**, i8**)* @repo to i64), i64 ptrtoint (%swift.async_func_pointer* @repoTu to i64)) to i32), i32 16 }>, align 8

declare token @llvm.coro.id.async(i32, i32, i32, i8*) #0

declare i8* @llvm.coro.begin(token, i8* writeonly) #0

declare i1 @llvm.coro.end.async(i8*, i1, ...) #0

define swifttailcc void @repo(%swift.context* swiftasync %0, %swift.opaque* noalias nocapture %1, %swift.opaque* noalias nocapture %2, i8* %3, %swift.refcounted* %4, %swift.type* %Self, i8** %Self.AsyncSequence, i8** %Self.Element.Comparable) #1 {
entry:
  %5 = alloca %swift.context*, align 8
  %6 = bitcast %swift.context* %0 to <{ %swift.context*, void (%swift.context*)* }>*
  %7 = call token @llvm.coro.id.async(i32 16, i32 16, i32 0, i8* bitcast (%swift.async_func_pointer* @repoTu to i8*))
  %8 = call i8* @llvm.coro.begin(token %7, i8* null)
  store %swift.context* %0, %swift.context** %5, align 8
  %9 = bitcast i8* %3 to i1 (%swift.opaque*, %swift.opaque*, %swift.refcounted*)*
  %10 = call swiftcc i1 %9(%swift.opaque* noalias nocapture %1, %swift.opaque* noalias nocapture %2, %swift.refcounted* swiftself %4) #2
  %11 = load %swift.context*, %swift.context** %5, align 8
  %12 = bitcast %swift.context* %11 to <{ %swift.context*, void (%swift.context*)* }>*
  %13 = getelementptr inbounds <{ %swift.context*, void (%swift.context*)* }>, <{ %swift.context*, void (%swift.context*)* }>* %12, i32 0, i32 1
  %14 = load void (%swift.context*)*, void (%swift.context*)** %13, align 8
  %15 = bitcast void (%swift.context*)* %14 to void (%swift.context*, i1, %swift.error*)*
  %16 = load %swift.context*, %swift.context** %5, align 8
  %17 = bitcast void (%swift.context*, i1, %swift.error*)* %15 to i8*
  %18 = call i1 (i8*, i1, ...) @llvm.coro.end.async(i8* %8, i1 false, void (i8*, %swift.context*, i1, %swift.error*)* @repo.0, i8* %17, %swift.context* %16, i1 %10, %swift.error* null)
  unreachable
}

; CHECK: define swifttailcc void @repo
; CHECK-NOT:llvm.coro.end.async
; CHECK: musttail call swifttailcc void
; CHECK-NOT:llvm.coro.end.async
; CHECK-NOT: unreachable
; CHECK: ret

define internal swifttailcc void @repo.0(i8* %0, %swift.context* %1, i1 %2, %swift.error* %3) #0 {
entry:
  %4 = bitcast i8* %0 to void (%swift.context*, i1, %swift.error*)*
  musttail call swifttailcc void %4(%swift.context* swiftasync %1, i1 %2, %swift.error* swiftself %3)
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind }
