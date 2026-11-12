; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; This testcase is invalid because we are indexing into a vector
; with non-byte-addressable element type (i1).

; CHECK-DAG: GEP into vector with non-byte-addressable element type
; CHECK-DAG: ptr getelementptr (<2 x i1>, ptr @x, i64 0, i64 1)
@x = global i8 0
@p = global ptr getelementptr (<2 x i1>, ptr @x, i64 0, i64 1)

; CHECK-DAG: GEP into vector with non-byte-addressable element type
; CHECK-DAG: %gep = getelementptr <2 x i1>, ptr %p, i64 0, i64 1
define ptr @test(ptr %p) {
  %gep = getelementptr <2 x i1>, ptr %p, i64 0, i64 1
  ret ptr %gep
}
