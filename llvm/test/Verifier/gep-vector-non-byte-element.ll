; RUN: not llvm-as < %s 2>&1 | FileCheck %s
; CHECK: GEP into vector with non-byte-addressable element type

; This testcase is invalid because we are indexing into a vector
; with non-byte-addressable element type (i1).

define ptr @test(ptr %p) {
  %gep = getelementptr <2 x i1>, ptr %p, i64 0, i64 1
  ret ptr %gep
}
