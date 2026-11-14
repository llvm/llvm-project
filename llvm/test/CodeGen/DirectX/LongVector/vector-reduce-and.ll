; RUN: llc -mtriple=dxil-pc-shadermodel6.8-library -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-SCALAR
; RUN: llc -mtriple=dxil-pc-shadermodel6.9-library -stop-before=dxil-op-lower -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-VECTOR

; CHECK-LABEL: define i32 @test_vector_reduce_and(
; CHECK-SCALAR-COUNT-16: and i32
; CHECK-VECTOR: call i32 @llvm.vector.reduce.and.v17i32(<17 x i32> %a)
define i32 @test_vector_reduce_and(<17 x i32> %a) {
  %result = call i32 @llvm.vector.reduce.and.v17i32(<17 x i32> %a)
  ret i32 %result
}

; CHECK-LABEL: define i32 @test_vector_reduce_andV4(
; CHECK-SCALAR-COUNT-3: and i32
; CHECK-VECTOR: call i32 @llvm.vector.reduce.and.v4i32(<4 x i32> %a)
define i32 @test_vector_reduce_andV4(<4 x i32> %a) {
  %result = call i32 @llvm.vector.reduce.and.v4i32(<4 x i32> %a)
  ret i32 %result
}
