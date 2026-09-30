// RUN: mlir-opt --split-input-file -pass-pipeline="builtin.module(func.func(tosa-to-linalg-named))" %s -verify-diagnostics

func.func @max_pool2d_adaptive_not_canonicalized(
    %arg0: tensor<1x4x4x1xf32>) -> tensor<1x2x2x1xf32> {
  %kernel = tosa.const_shape values(dense<[2, 2]> : tensor<2xindex>) : () -> !tosa.shape<2>
  %stride = tosa.const_shape values(dense<[2, 2]> : tensor<2xindex>) : () -> !tosa.shape<2>
  %pad = tosa.const_shape values(dense<[0, 0, 0, 0]> : tensor<4xindex>) : () -> !tosa.shape<4>
  // expected-error@+1 {{failed to legalize operation 'tosa.max_pool2d_adaptive'}}
  %0 = tosa.max_pool2d_adaptive %arg0, %kernel, %stride, %pad nan_mode<IGNORE> :
    (tensor<1x4x4x1xf32>, !tosa.shape<2>, !tosa.shape<2>, !tosa.shape<4>) -> tensor<1x2x2x1xf32>
  return %0 : tensor<1x2x2x1xf32>
}
