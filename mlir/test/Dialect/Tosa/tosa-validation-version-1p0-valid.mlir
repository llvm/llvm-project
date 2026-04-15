// RUN: mlir-opt %s -split-input-file -verify-diagnostics -tosa-attach-target="specification_version=1.0 profiles=pro_int,pro_fp" -tosa-validate="strict-op-spec-alignment" | FileCheck %s

// CHECK-LABEL: test_gather_i8_i32
func.func @test_gather_i8_i32(%input: tensor<13x27x3xi8>, %indices: tensor<13x26xi32>) -> tensor<13x26x3xi8> {
  %gather = tosa.gather %input, %indices : (tensor<13x27x3xi8>, tensor<13x26xi32>) -> tensor<13x26x3xi8>
  return %gather : tensor<13x26x3xi8>
}

// -----

// CHECK-LABEL: test_scatter_i8_i32
func.func @test_scatter_i8_i32(%input: tensor<13x27x3xi8>, %indices: tensor<13x26xi32>, %updates: tensor<13x26x3xi8>) -> tensor<13x27x3xi8> {
  %scatter = tosa.scatter %input, %indices, %updates : (tensor<13x27x3xi8>, tensor<13x26xi32>, tensor<13x26x3xi8>) -> tensor<13x27x3xi8>
  return %scatter : tensor<13x27x3xi8>
}
