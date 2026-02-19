// RUN: mlir-opt %s -emit-bytecode | mlir-opt | FileCheck %s

func.func @test_i1_splat_true() -> tensor<100xi1> {
  %0 = arith.constant dense<true> : tensor<100xi1>
  return %0 : tensor<100xi1>
}

// CHECK-LABEL: func.func @test_i1_splat_true
// CHECK: arith.constant dense<true> : tensor<100xi1>

func.func @test_i1_splat_false() -> tensor<100xi1> {
  %0 = arith.constant dense<false> : tensor<100xi1>
  return %0 : tensor<100xi1>
}

// CHECK-LABEL: func.func @test_i1_splat_false
// CHECK: arith.constant dense<false> : tensor<100xi1>
