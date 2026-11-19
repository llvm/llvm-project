; RUN: opt < %s -passes=dse -S | FileCheck %s

declare noalias ptr @malloc(i64) allockind("alloc,uninitialized")
declare noalias ptr @calloc(i64, i64) allockind("alloc,zeroed")
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)

define ptr @test_dse_malloc_memset_to_calloc() {
; CHECK-LABEL: define ptr @test_dse_malloc_memset_to_calloc()
; CHECK-NEXT: %calloc = call ptr @calloc(i64 1, i64 16), !alloc_token [[META:![0-9]+]]
; CHECK-NEXT: ret ptr %calloc
  %1 = tail call ptr @malloc(i64 16), !alloc_token !0
  call void @llvm.memset.p0.i64(ptr %1, i8 0, i64 16, i1 false)
  ret ptr %1
}

!0 = !{!"StructA", i1 true}
; CHECK: [[META]] = !{!"StructA", i1 true}
