// RUN: mlir-opt %s -split-input-file -verify-diagnostics -tosa-attach-target="specification_version=1.1.draft profiles=pro_fp" -tosa-validate="strict-op-spec-alignment"

// -----

func.func @test_avg_pool2d_adaptive_non_const_input_zp(%arg0: tensor<1x32x32x8xf32>, %input_zp: tensor<1xf32>) -> tensor<1x32x32x8xf32> {
  %output_zp = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
  %kernel = tosa.const_shape {values = dense<[1, 1]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %stride = tosa.const_shape {values = dense<[1, 1]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %pad = tosa.const_shape {values = dense<[0, 0, 0, 0]> : tensor<4xindex>} : () -> !tosa.shape<4>
  // expected-error@+1 {{'tosa.avg_pool2d_adaptive' op expected compile time resolvable constant, but got variable value for operand #1}}
  %0 = "tosa.avg_pool2d_adaptive"(%arg0, %input_zp, %output_zp, %kernel, %stride, %pad) {acc_type = f32} :
       (tensor<1x32x32x8xf32>, tensor<1xf32>, tensor<1xf32>, !tosa.shape<2>, !tosa.shape<2>, !tosa.shape<4>) -> tensor<1x32x32x8xf32>
  return %0 : tensor<1x32x32x8xf32>
}

// -----

func.func @test_avg_pool2d_adaptive_non_const_output_zp(%arg0: tensor<1x32x32x8xf32>, %output_zp: tensor<1xf32>) -> tensor<1x32x32x8xf32> {
  %input_zp = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
  %kernel = tosa.const_shape {values = dense<[1, 1]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %stride = tosa.const_shape {values = dense<[1, 1]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %pad = tosa.const_shape {values = dense<[0, 0, 0, 0]> : tensor<4xindex>} : () -> !tosa.shape<4>
  // expected-error@+1 {{'tosa.avg_pool2d_adaptive' op expected compile time resolvable constant, but got variable value for operand #2}}
  %0 = "tosa.avg_pool2d_adaptive"(%arg0, %input_zp, %output_zp, %kernel, %stride, %pad) {acc_type = f32} :
       (tensor<1x32x32x8xf32>, tensor<1xf32>, tensor<1xf32>, !tosa.shape<2>, !tosa.shape<2>, !tosa.shape<4>) -> tensor<1x32x32x8xf32>
  return %0 : tensor<1x32x32x8xf32>
}
