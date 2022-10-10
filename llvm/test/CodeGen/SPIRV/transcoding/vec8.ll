;; This test verifies that the Vector16 capability is correctly added
;; if an OpenCL kernel uses a vector of eight elements.
;;
;; Source:
;; __kernel void test( int8 v ) {}

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV: OpCapability Vector16

define spir_kernel void @test(<8 x i32> %v) {
  %1 = alloca <8 x i32>, align 32
  store <8 x i32> %v, <8 x i32>* %1, align 32
  ret void
}
