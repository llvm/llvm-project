; RUN: opt < %s -instcombine -S | FileCheck %s
; PR1201
define i32 @main(i32 %argc, i8** %argv) {
; CHECK: @main
    %c_19 = alloca i8*
    %malloc_206 = tail call i8* @malloc(i32 mul (i32 ptrtoint (i8* getelementptr (i8* null, i32 1) to i32), i32 10))
    store i8* %malloc_206, i8** %c_19
    %tmp_207 = load i8** %c_19
    tail call void @free(i8* %tmp_207)
    ret i32 0
; CHECK-NEXT: ret i32 0
}

declare noalias i8* @calloc(i32, i32) nounwind
declare noalias i8* @malloc(i32)
declare void @free(i8*)

define i1 @foo() {
; CHECK: @foo
; CHECK-NEXT: ret i1 false
  %m = call i8* @malloc(i32 1)
  %z = icmp eq i8* %m, null
  call void @free(i8* %m)
  ret i1 %z
}

declare void @llvm.lifetime.start(i64, i8*)
declare void @llvm.lifetime.end(i64, i8*)
declare i64 @llvm.objectsize.i64(i8*, i1)
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind
declare void @llvm.memmove.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind
declare void @llvm.memset.p0i8.i32(i8*, i8, i32, i32, i1) nounwind

define void @test3(i8* %src) {
; CHECK: @test3
; CHECK-NEXT: ret void
  %a = call noalias i8* @malloc(i32 10)
  call void @llvm.lifetime.start(i64 10, i8* %a)
  call void @llvm.lifetime.end(i64 10, i8* %a)
  %size = call i64 @llvm.objectsize.i64(i8* %a, i1 true)
  store i8 42, i8* %a
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %a, i8* %src, i32 32, i32 1, i1 false)
  call void @llvm.memmove.p0i8.p0i8.i32(i8* %a, i8* %src, i32 32, i32 1, i1 false)
  call void @llvm.memset.p0i8.i32(i8* %a, i8 5, i32 32, i32 1, i1 false)
  %alloc2 = call noalias i8* @calloc(i32 5, i32 7) nounwind
  %z = icmp ne i8* %alloc2, null
  ret void
}

;; This used to crash.
define void @test4() {
; CHECK: @test4
; CHECK-NEXT: ret void
  %A = call i8* @malloc(i32 16000)
  %B = bitcast i8* %A to double*
  %C = bitcast double* %B to i8*
  call void @free(i8* %C)
  ret void
}

; CHECK: @test5
define void @test5(i8* %ptr, i8** %esc) {
; CHECK-NEXT: call i8* @malloc
; CHECK-NEXT: call i8* @malloc
; CHECK-NEXT: call i8* @malloc
; CHECK-NEXT: call i8* @malloc
; CHECK-NEXT: call i8* @malloc
; CHECK-NEXT: call i8* @malloc
; CHECK-NEXT: call i8* @malloc
; CHECK-NEXT: call void @llvm.memcpy
; CHECK-NEXT: call void @llvm.memmove
; CHECK-NEXT: store
; CHECK-NEXT: call void @llvm.memcpy
; CHECK-NEXT: call void @llvm.memmove
; CHECK-NEXT: call void @llvm.memset
; CHECK-NEXT: store volatile
; CHECK-NEXT: ret
  %a = call i8* @malloc(i32 700)
  %b = call i8* @malloc(i32 700)
  %c = call i8* @malloc(i32 700)
  %d = call i8* @malloc(i32 700)
  %e = call i8* @malloc(i32 700)
  %f = call i8* @malloc(i32 700)
  %g = call i8* @malloc(i32 700)
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %ptr, i8* %a, i32 32, i32 1, i1 false)
  call void @llvm.memmove.p0i8.p0i8.i32(i8* %ptr, i8* %b, i32 32, i32 1, i1 false)
  store i8* %c, i8** %esc
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %d, i8* %ptr, i32 32, i32 1, i1 true)
  call void @llvm.memmove.p0i8.p0i8.i32(i8* %e, i8* %ptr, i32 32, i32 1, i1 true)
  call void @llvm.memset.p0i8.i32(i8* %f, i8 5, i32 32, i32 1, i1 true)
  store volatile i8 4, i8* %g
  ret void
}
