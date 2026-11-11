// RUN: mlir-opt %s -split-input-file -pass-pipeline="builtin.module(func.func(tosa-to-linalg))" -verify-diagnostics
// RUN: mlir-opt %s -split-input-file -tosa-to-linalg-pipeline="validation=false" -verify-diagnostics

// Without tosa-validate, unsupported multiples must fail conversion rather than
// construct invalid tensor types. Return the tile result so it is not removed
// by the pipeline's canonicalizer.
func.func @tile_negative_multiple(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %multiples = tosa.const_shape values(dense<-2> : tensor<1xindex>) : () -> !tosa.shape<1>
  // expected-error@+1 {{failed to legalize operation 'tosa.tile'}}
  %0 = tosa.tile %arg0, %multiples : (tensor<4xf32>, !tosa.shape<1>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

func.func @tile_zero_multiple(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %multiples = tosa.const_shape values(dense<0> : tensor<1xindex>) : () -> !tosa.shape<1>
  // expected-error@+1 {{failed to legalize operation 'tosa.tile'}}
  %0 = tosa.tile %arg0, %multiples : (tensor<4xf32>, !tosa.shape<1>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

func.func @tile_negative_multiple_dynamic_input(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %multiples = tosa.const_shape values(dense<-2> : tensor<1xindex>) : () -> !tosa.shape<1>
  // expected-error@+1 {{failed to legalize operation 'tosa.tile'}}
  %0 = tosa.tile %arg0, %multiples : (tensor<?xf32>, !tosa.shape<1>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

func.func @tile_minimum_multiple(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %multiples = tosa.const_shape values(dense<-9223372036854775808> : tensor<1xindex>) : () -> !tosa.shape<1>
  // expected-error@+1 {{failed to legalize operation 'tosa.tile'}}
  %0 = tosa.tile %arg0, %multiples : (tensor<4xf32>, !tosa.shape<1>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// A tile with invalid multiples must not be eliminated as a broadcast before
// conversion, even when its input has a singleton dimension.
func.func @tile_broadcast_negative_multiple(%arg0: tensor<1xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %multiples = tosa.const_shape values(dense<-2> : tensor<1xindex>) : () -> !tosa.shape<1>
  // expected-error@+1 {{failed to legalize operation 'tosa.tile'}}
  %tile = tosa.tile %arg0, %multiples : (tensor<1xf32>, !tosa.shape<1>) -> tensor<4xf32>
  %add = tosa.add %tile, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %add : tensor<4xf32>
}

// -----

func.func @tile_broadcast_zero_multiple(%arg0: tensor<1xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %multiples = tosa.const_shape values(dense<0> : tensor<1xindex>) : () -> !tosa.shape<1>
  // expected-error@+1 {{failed to legalize operation 'tosa.tile'}}
  %tile = tosa.tile %arg0, %multiples : (tensor<1xf32>, !tosa.shape<1>) -> tensor<4xf32>
  %add = tosa.add %tile, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %add : tensor<4xf32>
}
