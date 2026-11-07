; RUN: opt -passes=nary-reassociate -S < %s | FileCheck %s

@a = global [16 x i32] zeroinitializer

define ptr @test_reassociate_no_inbounds(i64 %i, i64 %j, ptr %out) {
; CHECK-LABEL: @test_reassociate_no_inbounds(
; CHECK: getelementptr i32, ptr %base, i64 %j
; CHECK-NOT: getelementptr inbounds i32, ptr %base
  %base = getelementptr i32, ptr @a, i64 %i
  store ptr %base, ptr %out, align 8
  %ij = add i64 %i, %j
  %gep = getelementptr inbounds i32, ptr @a, i64 %ij
  ret ptr %gep
}

