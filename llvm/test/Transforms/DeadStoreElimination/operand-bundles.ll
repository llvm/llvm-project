; RUN: opt < %s -passes=dse -S | FileCheck %s

declare noalias ptr @malloc(i64) "malloc-like"

declare void @foo()
declare void @bar(ptr)

define void @test() {
  %obj = call ptr @malloc(i64 8)
  store i8 0, ptr %obj
  ; don't remove store. %obj should be treated like it will be read by the @foo.
  ; CHECK: store i8 0, ptr %obj
  call void @foo() ["deopt" (ptr %obj)]
  ret void
}

define void @test1() {
  %obj = call ptr @malloc(i64 8)
  store i8 0, ptr %obj
  ; CHECK: store i8 0, ptr %obj
  call void @bar(ptr nocapture %obj)
  ret void
}

define void @test2() {
  %obj = call ptr @malloc(i64 8)
  store i8 0, ptr %obj
  ; CHECK-NOT: store i8 0, ptr %obj
  call void @foo()
  ret void
}

define void @test3() {
  ; CHECK-LABEL: @test3(
  %s = alloca i64
  ; Verify that this first store is not considered killed by the second one
  ; since it could be observed from the deopt continuation.
  ; CHECK: store i64 1, ptr %s
  store i64 1, ptr %s
  call void @foo() [ "deopt"(ptr %s) ]
  store i64 0, ptr %s
  ret void
}

declare noalias ptr @calloc(i64, i64) inaccessiblememonly allockind("alloc,zeroed")

define void @test4() {
; CHECK-LABEL: @test4
  %local_obj = call ptr @calloc(i64 1, i64 4)
  call void @foo() ["deopt" (ptr %local_obj)]
  store i8 0, ptr %local_obj, align 4
  ; CHECK-NOT: store i8 0, ptr %local_obj, align 4
  call void @bar(ptr nocapture %local_obj)
  ret void
}
