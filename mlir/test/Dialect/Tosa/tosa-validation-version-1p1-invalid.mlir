// RUN: mlir-opt %s -split-input-file -verify-diagnostics -tosa-attach-target="specification_version=1.1.draft profiles=pro_fp extensions=mx_common,mx_fp4e2m1" -tosa-validate="strict-op-spec-alignment"

// -----

func.func @test_avg_pool2d_adaptive_non_const_input_zp(%arg0: tensor<1x32x32x8xf32>, %input_zp: tensor<1xf32>) -> tensor<1x32x32x8xf32> {
  %output_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %kernel = tosa.const_shape values(dense<[1, 1]> : tensor<2xindex>) : () -> !tosa.shape<2>
  %stride = tosa.const_shape values(dense<[1, 1]> : tensor<2xindex>) : () -> !tosa.shape<2>
  %pad = tosa.const_shape values(dense<[0, 0, 0, 0]> : tensor<4xindex>) : () -> !tosa.shape<4>
  // expected-error@+1 {{'tosa.avg_pool2d_adaptive' op expected compile time resolvable constant, but got variable value for operand #1}}
  %0 = "tosa.avg_pool2d_adaptive"(%arg0, %input_zp, %output_zp, %kernel, %stride, %pad) <{acc_type = f32}> :
       (tensor<1x32x32x8xf32>, tensor<1xf32>, tensor<1xf32>, !tosa.shape<2>, !tosa.shape<2>, !tosa.shape<4>) -> tensor<1x32x32x8xf32>
  return %0 : tensor<1x32x32x8xf32>
}

// -----

func.func @test_avg_pool2d_adaptive_non_const_output_zp(%arg0: tensor<1x32x32x8xf32>, %output_zp: tensor<1xf32>) -> tensor<1x32x32x8xf32> {
  %input_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %kernel = tosa.const_shape values(dense<[1, 1]> : tensor<2xindex>) : () -> !tosa.shape<2>
  %stride = tosa.const_shape values(dense<[1, 1]> : tensor<2xindex>) : () -> !tosa.shape<2>
  %pad = tosa.const_shape values(dense<[0, 0, 0, 0]> : tensor<4xindex>) : () -> !tosa.shape<4>
  // expected-error@+1 {{'tosa.avg_pool2d_adaptive' op expected compile time resolvable constant, but got variable value for operand #2}}
  %0 = "tosa.avg_pool2d_adaptive"(%arg0, %input_zp, %output_zp, %kernel, %stride, %pad) <{acc_type = f32}> :
       (tensor<1x32x32x8xf32>, tensor<1xf32>, tensor<1xf32>, !tosa.shape<2>, !tosa.shape<2>, !tosa.shape<4>) -> tensor<1x32x32x8xf32>
  return %0 : tensor<1x32x32x8xf32>
}

// -----

module {
  // expected-error@+1 {{'tosa.variable' op is not profile-aligned: element type 'f64' is not legal}}
  tosa.variable @dynamic_var : tensor<2x?xf64>
}

// -----

func.func @test_slice_block_scaled_invalid_block_size(%arg0: tensor<4x4x64x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f4E2M1FN>>) -> tensor<1x1x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f4E2M1FN>> {
  %start = tosa.const_shape values(dense<[0, 0, 1]> : tensor<3xindex>) : () -> !tosa.shape<3>
  %size = tosa.const_shape values(dense<[1, 1, 32]> : tensor<3xindex>) : () -> !tosa.shape<3>
  // expected-error@+1 {{'tosa.slice' op expected start innermost block size to match data type for block scaled input, got start block=1, scale block=32}}
  %0 = tosa.slice %arg0, %start, %size {input_unsigned = false} : (tensor<4x4x64x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f4E2M1FN>>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<1x1x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f4E2M1FN>>
  return %0 : tensor<1x1x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f4E2M1FN>>
}
