; RUN: llc -mtriple=dxil-pc-shadermodel6.8-library -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-SCALAR
; RUN: llc -mtriple=dxil-pc-shadermodel6.9-library -stop-before=dxil-op-lower -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-VECTOR

; CHECK-LABEL: define i1 @test_any(
; CHECK-SCALAR-COUNT-17: icmp ne i32 {{.*}}, 0
; CHECK-SCALAR-COUNT-16: or i1
; CHECK-VECTOR: [[CMP:%.*]] = icmp ne <17 x i32> %a, zeroinitializer
; CHECK-VECTOR: call i1 @llvm.vector.reduce.or.v17i1(<17 x i1> [[CMP]])
define i1 @test_any(<17 x i32> %a) {
  %result = call i1 @llvm.dx.any.v17i32(<17 x i32> %a)
  ret i1 %result
}

; CHECK-LABEL: define i1 @test_any_float(
; CHECK-SCALAR-COUNT-17: fcmp une float {{.*}}, 0.000000e+00
; CHECK-SCALAR-COUNT-16: or i1
; CHECK-VECTOR: [[CMP:%.*]] = fcmp une <17 x float> %a, zeroinitializer
; CHECK-VECTOR: call i1 @llvm.vector.reduce.or.v17i1(<17 x i1> [[CMP]])
define i1 @test_any_float(<17 x float> %a) {
  %result = call i1 @llvm.dx.any.v17f32(<17 x float> %a)
  ret i1 %result
}
