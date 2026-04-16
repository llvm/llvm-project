// RUN: mlir-opt %s -split-input-file -verify-diagnostics -tosa-attach-target="specification_version=1.1.draft profiles=pro_fp" -tosa-validate="strict-op-spec-alignment" | FileCheck %s

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

// -----

// CHECK-LABEL: test_row_gather_block_scaled_i8_i32
func.func @test_row_gather_block_scaled_i8_i32(%input: tensor<13x27x3xi8>, %indices: tensor<13x26xi32>) -> tensor<13x52x3xi8> {
  %row_count = "tosa.const"() {values = dense<2> : tensor<1xi32>} : () -> tensor<1xi32>
  %gather = tosa.row_gather_block_scaled %input, %indices, %row_count {block_size = #tosa.block_size<BLOCK_SIZE_1>} : (tensor<13x27x3xi8>, tensor<13x26xi32>, tensor<1xi32>) -> (tensor<13x52x3xi8>)
  return %gather : tensor<13x52x3xi8>
}
