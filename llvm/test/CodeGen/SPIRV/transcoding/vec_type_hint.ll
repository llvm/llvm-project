; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

;; kernel
;; __attribute__((vec_type_hint(float4)))
;; void test_float() {}

;; kernel
;; __attribute__((vec_type_hint(double)))
;; void test_double() {}

;; kernel
;; __attribute__((vec_type_hint(uint4)))
;; void test_uint() {}

;; kernel
;; __attribute__((vec_type_hint(int8)))
;; void test_int() {}

; CHECK-SPIRV: OpEntryPoint {{.*}} %[[#]] "test_float"
; CHECK-SPIRV: OpEntryPoint {{.*}} %[[#]] "test_double"
; CHECK-SPIRV: OpEntryPoint {{.*}} %[[#]] "test_uint"
; CHECK-SPIRV: OpEntryPoint {{.*}} %[[#]] "test_int"
; CHECK-SPIRV: OpExecutionMode %[[#]] VecTypeHint [[#]]
; CHECK-SPIRV: OpExecutionMode %[[#]] VecTypeHint [[#]]
; CHECK-SPIRV: OpExecutionMode %[[#]] VecTypeHint [[#]]
; CHECK-SPIRV: OpExecutionMode %[[#]] VecTypeHint [[#]]

define dso_local spir_kernel void @test_float() !vec_type_hint !4 {
entry:
  ret void
}

define dso_local spir_kernel void @test_double() !vec_type_hint !5 {
entry:
  ret void
}

define dso_local spir_kernel void @test_uint() !vec_type_hint !6 {
entry:
  ret void
}

define dso_local spir_kernel void @test_int() !vec_type_hint !7 {
entry:
  ret void
}

!4 = !{<4 x float> undef, i32 0}
!5 = !{double undef, i32 0}
!6 = !{<4 x i32> undef, i32 0}
!7 = !{<8 x i32> undef, i32 1}
