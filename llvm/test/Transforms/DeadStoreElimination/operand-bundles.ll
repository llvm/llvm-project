; RUN: opt < %s -basicaa -dse -S | FileCheck %s

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
