; RUN: llvm-dis < %s.bc | FileCheck %s

@g = extern_weak global [32 x i8]

define ptr @test_scalable_vector_gep() {
; CHECK-LABEL: define ptr @test_scalable_vector_gep() {
; CHECK-NEXT: %constexpr = getelementptr <vscale x 1 x i8>, ptr @g, i64 1
; CHECK-NEXT: ret ptr %constexpr
  ret ptr getelementptr (<vscale x 1 x i8>, ptr @g, i64 1)
}
