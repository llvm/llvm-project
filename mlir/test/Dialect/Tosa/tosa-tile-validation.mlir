// RUN: mlir-opt %s -split-input-file -o /dev/null
// RUN: mlir-opt %s -split-input-file -verify-diagnostics -tosa-attach-target="level=none specification_version=1.1.draft profiles=pro_fp extensions=dynamic,shape" -tosa-validate="validate-function-signature=false"
// RUN: mlir-opt %s -split-input-file -verify-diagnostics -tosa-attach-target="level=none specification_version=1.1.draft profiles=pro_fp extensions=dynamic,shape" -tosa-validate="strict-op-spec-alignment validate-function-signature=false"

// Constant multiples values are checked by tosa-validate, not the op verifier.
func.func @test_tile_invalid_multiples_value(%arg0: tensor<4x31xf32>) {
  %multiples = tosa.const_shape values(dense<[2, -2]> : tensor<2xindex>) : () -> !tosa.shape<2>
  // expected-error@+1 {{'tosa.tile' op multiples values must be > 0 or -1, got [2, -2]}}
  %0 = tosa.tile %arg0, %multiples : (tensor<4x31xf32>, !tosa.shape<2>) -> tensor<8x?xf32>
  return
}

// -----

func.func @test_tile_zero_multiple(%arg0: tensor<4x31xf32>) {
  %multiples = tosa.const_shape values(dense<[0, 2]> : tensor<2xindex>) : () -> !tosa.shape<2>
  // expected-error@+1 {{'tosa.tile' op multiples values must be > 0 or -1, got [0, 2]}}
  %0 = tosa.tile %arg0, %multiples : (tensor<4x31xf32>, !tosa.shape<2>) -> tensor<?x62xf32>
  return
}

// -----

// The migration preserves the existing allowance for -1.
func.func @test_tile_inferable_multiple(%arg0: tensor<4x31xf32>) {
  %multiples = tosa.const_shape values(dense<[2, -1]> : tensor<2xindex>) : () -> !tosa.shape<2>
  %0 = tosa.tile %arg0, %multiples : (tensor<4x31xf32>, !tosa.shape<2>) -> tensor<8x?xf32>
  return
}

// -----

func.func @test_tile_positive_multiples(%arg0: tensor<4x31xf32>) -> tensor<4x62xf32> {
  %multiples = tosa.const_shape values(dense<[1, 2]> : tensor<2xindex>) : () -> !tosa.shape<2>
  %0 = tosa.tile %arg0, %multiples : (tensor<4x31xf32>, !tosa.shape<2>) -> tensor<4x62xf32>
  return %0 : tensor<4x62xf32>
}

// -----

func.func @test_tile_nonconstant_multiples(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %multiples = tosa.dim %arg0 axis(0) : (tensor<?xf32>) -> !tosa.shape<1>
  %0 = tosa.tile %arg0, %multiples : (tensor<?xf32>, !tosa.shape<1>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
