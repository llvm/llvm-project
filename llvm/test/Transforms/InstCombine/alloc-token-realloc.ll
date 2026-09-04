; RUN: opt < %s -passes=instcombine -S | FileCheck %s

declare ptr @realloc(ptr allocptr, i64) allockind("realloc") allocsize(1)
declare noalias ptr @malloc(i64) allockind("alloc,uninitialized")

define ptr @test_realloc_null_alloc_token() {
; CHECK-LABEL: define ptr @test_realloc_null_alloc_token()
; CHECK-NEXT: %malloc = call dereferenceable_or_null(16) ptr @malloc(i64 16), !alloc_token [[META:![0-9]+]]
; CHECK-NEXT: ret ptr %malloc
  %call = call ptr @realloc(ptr null, i64 16), !alloc_token !0
  ret ptr %call
}

!0 = !{!"StructA", i1 true}
; CHECK: [[META]] = !{!"StructA", i1 true}
