; RUN: not llvm-as < %s 2>&1 | FileCheck %s
; Test the case of an invalid pointee type on a constant GEP

; CHECK: invalid base element for constant getelementptr

define ptr @test_scalable_vector_gep(ptr %a) {
  ret ptr getelementptr (<vscale x 1 x i8>, ptr @a, i64 1)
}
