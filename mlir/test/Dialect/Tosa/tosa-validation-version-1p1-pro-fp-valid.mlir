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

// CHECK-LABEL: test_row_gather_i8_i32
func.func @test_row_gather_i8_i32(%input: tensor<13x27x3xi8>, %indices: tensor<13x26xi32>) -> tensor<13x52x3xi8> {
  %row_count = "tosa.const"() <{values = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
  %gather = tosa.row_gather %input, %indices, %row_count : (tensor<13x27x3xi8>, tensor<13x26xi32>, tensor<1xi32>) -> tensor<13x52x3xi8>
  return %gather : tensor<13x52x3xi8>
}

// -----

// CHECK-LABEL: test_row_gather_block_scaled_i8_i32
func.func @test_row_gather_block_scaled_i8_i32(%input: tensor<13x27x3xi8>, %indices: tensor<13x26xi32>) -> tensor<13x52x3xi8> {
  %row_count = "tosa.const"() <{values = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
  %gather = tosa.row_gather_block_scaled %input, %indices, %row_count block_size<BLOCK_SIZE_1> : (tensor<13x27x3xi8>, tensor<13x26xi32>, tensor<1xi32>) -> (tensor<13x52x3xi8>)
  return %gather : tensor<13x52x3xi8>
}

// -----

// CHECK-LABEL: test_concat_i8
func.func @test_concat_i8(%arg0: tensor<13x21x3xi8>, %arg1: tensor<13x21x3xi8>) -> tensor<26x21x3xi8> {
  %0 = tosa.concat %arg0, %arg1 axis(0) : (tensor<13x21x3xi8>, tensor<13x21x3xi8>) -> tensor<26x21x3xi8>
  return %0 : tensor<26x21x3xi8>
}

// -----

// CHECK-LABEL: test_pad_i16
func.func @test_pad_i16(%arg0: tensor<13x21x3xi16>) -> tensor<13x21x3xi16> {
  %0 = "tosa.const"() <{values = dense<3> : tensor<1xi16>}> : () -> tensor<1xi16>
  %padding = tosa.const_shape values(dense<0> : tensor<6xindex>) : () -> !tosa.shape<6>
  %1 = tosa.pad %arg0, %padding, %0 : (tensor<13x21x3xi16>, !tosa.shape<6>, tensor<1xi16>) -> tensor<13x21x3xi16>
  return %1 : tensor<13x21x3xi16>
}

// -----

// CHECK-LABEL: test_reshape_i32
func.func @test_reshape_i32(%arg0: tensor<13x21x3xi32>) -> tensor<1x819xi32> {
  %1 = tosa.const_shape values(dense<[1, 819]> : tensor<2xindex>) : () -> !tosa.shape<2>
  %0 = tosa.reshape %arg0, %1 : (tensor<13x21x3xi32>, !tosa.shape<2>) -> tensor<1x819xi32>
  return %0 : tensor<1x819xi32>
}

// -----

// CHECK-LABEL: test_reverse_i8
func.func @test_reverse_i8(%arg0: tensor<13x21x3xi8>) -> tensor<13x21x3xi8> {
  %0 = tosa.reverse %arg0 axis(0) : (tensor<13x21x3xi8>) -> tensor<13x21x3xi8>
  return %0 : tensor<13x21x3xi8>
}

// -----

// CHECK-LABEL: test_slice_i16
func.func @test_slice_i16(%arg0: tensor<13x21x3xi16>) -> tensor<4x11x1xi16> {
  %size = tosa.const_shape values(dense<[4, 11, 1]> : tensor<3xindex>) : () -> !tosa.shape<3>
  %start = tosa.const_shape values(dense<[6, 8, 0]> : tensor<3xindex>) : () -> !tosa.shape<3>
  %2 = tosa.slice %arg0, %start, %size : (tensor<13x21x3xi16>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<4x11x1xi16>
  return %2 : tensor<4x11x1xi16>
}

// -----

// CHECK-LABEL: test_tile_i32
func.func @test_tile_i32(%arg0: tensor<13x21x3xi32>) -> tensor<39x21x6xi32> {
  %cst = tosa.const_shape values(dense<[3, 1, 2]> : tensor<3xindex>) : () -> !tosa.shape<3>
  %0 = tosa.tile %arg0, %cst: (tensor<13x21x3xi32>, !tosa.shape<3>) -> tensor<39x21x6xi32>
  return %0 : tensor<39x21x6xi32>
}

// -----

// CHECK-LABEL: test_transpose_i8
func.func @test_transpose_i8(%arg0: tensor<13x21x3xi8>) -> tensor<3x13x21xi8> {
  %1 = tosa.transpose %arg0 perms([2, 0, 1]) : (tensor<13x21x3xi8>) -> tensor<3x13x21xi8>
  return %1 : tensor<3x13x21xi8>
}