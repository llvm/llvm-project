; RUN: opt < %s -passes='function(coro-early),cgscc(coro-split)' -S | FileCheck %s
target datalayout = "E-p:64:64"

%swift.type = type { i64 }
%swift.opaque = type opaque
%T4red215EmptyCollectionV = type opaque
%TSi = type <{ i64 }>

define hidden swiftcc { i8*, %swift.opaque* } @no_suspends(i8* %buffer, i64 %arg) #1 {
  %id = call token @llvm.coro.id.retcon.once(i32 32, i32 8, i8* %buffer, i8* bitcast (void (i8*, i1)* @prototype to i8*), i8* bitcast (i8* (i64)* @malloc to i8*), i8* bitcast (void (i8*)* @free to i8*))
  %begin = call i8* @llvm.coro.begin(token %id, i8* null)
  call void @print(i64 %arg)
  call void @llvm.trap()
  unreachable

bb1:
  call void @print(i64 %arg)
  call i1 @llvm.coro.end(i8* %begin, i1 false)
  unreachable
}
; CHECK-LABEL: define hidden swiftcc { i8*, %swift.opaque* } @no_suspends(
; CHECK:         call token @llvm.coro.id.retcon.once
; CHECK-NEXT:    call void @print(i64 %arg)
; CHECK-NEXT:    call void @llvm.trap()
; CHECK-NEXT:    unreachable

declare swiftcc void @prototype(i8* noalias dereferenceable(32), i1)
declare void @print(i64)

declare noalias i8* @malloc(i64) #5
declare void @free(i8* nocapture) #5

declare token @llvm.coro.id.retcon.once(i32, i32, i8*, i8*, i8*, i8*) #5
declare i8* @llvm.coro.begin(token, i8* writeonly) #5
declare token @llvm.coro.alloca.alloc.i64(i64, i32) #5
declare i8* @llvm.coro.alloca.get(token) #5
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #6
declare i1 @llvm.coro.suspend.retcon.i1(...) #5
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #6
declare void @llvm.coro.alloca.free(token) #5
declare i1 @llvm.coro.end(i8*, i1) #5

declare void @llvm.trap()

attributes #1 = { noreturn nounwind }
attributes #5 = { nounwind }
