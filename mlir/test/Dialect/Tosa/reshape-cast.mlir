// RUN: mlir-opt --split-input-file --canonicalize="test-convergence" %s | FileCheck %s

// CHECK-LABEL: @inverse_reshapes_bf16
func.func @inverse_reshapes_bf16(%arg0: tensor<1x4x8xf32>) -> tensor<1x4x8xbf16> {
  // CHECK-NEXT: %[[CAST:.*]] = tosa.cast %arg0 {test.tag = "keep"} : (tensor<1x4x8xf32>) -> tensor<1x4x8xbf16>
  // CHECK-NEXT: return %[[CAST]] : tensor<1x4x8xbf16>
  %s2 = tosa.const_shape {values = dense<[4, 8]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %s3 = tosa.const_shape {values = dense<[1, 4, 8]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %r = tosa.reshape %arg0, %s2 : (tensor<1x4x8xf32>, !tosa.shape<2>) -> tensor<4x8xf32>
  %c = tosa.cast %r {test.tag = "keep"} : (tensor<4x8xf32>) -> tensor<4x8xbf16>
  %o = tosa.reshape %c, %s3 : (tensor<4x8xbf16>, !tosa.shape<3>) -> tensor<1x4x8xbf16>
  return %o : tensor<1x4x8xbf16>
}

// -----

// CHECK-LABEL: @inverse_reshapes_integer
func.func @inverse_reshapes_integer(%arg0: tensor<2x4xi8>) -> tensor<2x4xi32> {
  // CHECK-NEXT: %[[CAST:.*]] = tosa.cast %arg0 : (tensor<2x4xi8>) -> tensor<2x4xi32>
  // CHECK-NEXT: return %[[CAST]] : tensor<2x4xi32>
  %s1 = tosa.const_shape {values = dense<8> : tensor<1xindex>} : () -> !tosa.shape<1>
  %s2 = tosa.const_shape {values = dense<[2, 4]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %r = tosa.reshape %arg0, %s1 : (tensor<2x4xi8>, !tosa.shape<1>) -> tensor<8xi8>
  %c = tosa.cast %r : (tensor<8xi8>) -> tensor<8xi32>
  %o = tosa.reshape %c, %s2 : (tensor<8xi32>, !tosa.shape<2>) -> tensor<2x4xi32>
  return %o : tensor<2x4xi32>
}

// -----

// CHECK-LABEL: @different_output_shape
func.func @different_output_shape(%arg0: tensor<1x4x8xf32>) -> tensor<2x16xbf16> {
  // CHECK: %[[RESHAPE:.*]] = tosa.reshape %arg0
  // CHECK: %[[CAST:.*]] = tosa.cast %[[RESHAPE]]
  // CHECK: %[[OUTPUT:.*]] = tosa.reshape %[[CAST]]
  // CHECK: return %[[OUTPUT]]
  %s1 = tosa.const_shape {values = dense<32> : tensor<1xindex>} : () -> !tosa.shape<1>
  %s2 = tosa.const_shape {values = dense<[2, 16]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %r = tosa.reshape %arg0, %s1 : (tensor<1x4x8xf32>, !tosa.shape<1>) -> tensor<32xf32>
  %c = tosa.cast %r : (tensor<32xf32>) -> tensor<32xbf16>
  %o = tosa.reshape %c, %s2 : (tensor<32xbf16>, !tosa.shape<2>) -> tensor<2x16xbf16>
  return %o : tensor<2x16xbf16>
}

// -----

// CHECK-LABEL: @shared_cast
func.func @shared_cast(%arg0: tensor<1x4x8xf32>) -> (tensor<4x8xbf16>, tensor<1x4x8xbf16>) {
  // CHECK: %[[RESHAPE:.*]] = tosa.reshape %arg0
  // CHECK: %[[CAST:.*]] = tosa.cast %[[RESHAPE]]
  // CHECK-NOT: tosa.cast
  // CHECK: %[[OUTPUT:.*]] = tosa.reshape %[[CAST]]
  // CHECK-NEXT: return %[[CAST]], %[[OUTPUT]]
  %s2 = tosa.const_shape {values = dense<[4, 8]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %s3 = tosa.const_shape {values = dense<[1, 4, 8]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %r = tosa.reshape %arg0, %s2 : (tensor<1x4x8xf32>, !tosa.shape<2>) -> tensor<4x8xf32>
  %c = tosa.cast %r : (tensor<4x8xf32>) -> tensor<4x8xbf16>
  %o = tosa.reshape %c, %s3 : (tensor<4x8xbf16>, !tosa.shape<3>) -> tensor<1x4x8xbf16>
  return %c, %o : tensor<4x8xbf16>, tensor<1x4x8xbf16>
}

// -----

// CHECK-LABEL: @dynamic_shapes
func.func @dynamic_shapes(%arg0: tensor<1x?x4xf32>) -> tensor<1x?x4xbf16> {
  // CHECK: %[[RESHAPE:.*]] = tosa.reshape %arg0
  // CHECK: %[[CAST:.*]] = tosa.cast %[[RESHAPE]]
  // CHECK: %[[OUTPUT:.*]] = tosa.reshape %[[CAST]]
  // CHECK: return %[[OUTPUT]]
  %s2 = tosa.const_shape {values = dense<[-1, 4]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %s3 = tosa.const_shape {values = dense<[1, -1, 4]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %r = tosa.reshape %arg0, %s2 : (tensor<1x?x4xf32>, !tosa.shape<2>) -> tensor<?x4xf32>
  %c = tosa.cast %r : (tensor<?x4xf32>) -> tensor<?x4xbf16>
  %o = tosa.reshape %c, %s3 : (tensor<?x4xbf16>, !tosa.shape<3>) -> tensor<1x?x4xbf16>
  return %o : tensor<1x?x4xbf16>
}

// -----

// CHECK-LABEL: @block_scaled_cast
func.func @block_scaled_cast(%arg0: tensor<64x32xf32>) -> tensor<64x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>> {
  // CHECK: %[[RESHAPE:.*]] = tosa.reshape %arg0
  // CHECK: %[[CAST:.*]] = tosa.cast %[[RESHAPE]]
  // CHECK: %[[OUTPUT:.*]] = tosa.reshape %[[CAST]]
  // CHECK: return %[[OUTPUT]]
  %s = tosa.const_shape {values = dense<[32, 64]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %original = tosa.const_shape {values = dense<[64, 32]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %r = tosa.reshape %arg0, %s : (tensor<64x32xf32>, !tosa.shape<2>) -> tensor<32x64xf32>
  %c = tosa.cast %r : (tensor<32x64xf32>) -> tensor<32x64x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>
  %o = tosa.reshape %c, %original : (tensor<32x64x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>, !tosa.shape<2>) -> tensor<64x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>
  return %o : tensor<64x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>
}
