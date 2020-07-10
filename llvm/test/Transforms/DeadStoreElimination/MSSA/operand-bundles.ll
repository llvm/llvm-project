; RUN: opt < %s -basic-aa -dse -enable-dse-memoryssa -S | FileCheck %s

declare noalias i8* @malloc(i64) "malloc-like"

declare void @foo()
declare void @bar(i8*)

define void @test() {
  %obj = call i8* @malloc(i64 8)
  store i8 0, i8* %obj
  ; don't remove store. %obj should be treated like it will be read by the @foo.
  ; CHECK: store i8 0, i8* %obj
  call void @foo() ["deopt" (i8* %obj)]
  ret void
}

define void @test1() {
  %obj = call i8* @malloc(i64 8)
  store i8 0, i8* %obj
  ; CHECK: store i8 0, i8* %obj
  call void @bar(i8* nocapture %obj)
  ret void
}

define void @test2() {
  %obj = call i8* @malloc(i64 8)
  store i8 0, i8* %obj
  ; CHECK-NOT: store i8 0, i8* %obj
  call void @foo()
  ret void
}

define void @test3() {
  ; CHECK-LABEL: @test3(
  %s = alloca i64
  ; Verify that this first store is not considered killed by the second one
  ; since it could be observed from the deopt continuation.
  ; CHECK: store i64 1, i64* %s
  store i64 1, i64* %s
  call void @foo() [ "deopt"(i64* %s) ]
  store i64 0, i64* %s
  ret void
}

declare noalias i8* @calloc(i64, i64)

define void @test4() {
; CHECK-LABEL: @test4
  %local_obj = call i8* @calloc(i64 1, i64 4)
  call void @foo() ["deopt" (i8* %local_obj)]
  store i8 0, i8* %local_obj, align 4
  ; CHECK-NOT: store i8 0, i8* %local_obj, align 4
  call void @bar(i8* nocapture %local_obj)
  ret void
}
