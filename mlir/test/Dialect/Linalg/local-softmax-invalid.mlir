// RUN: mlir-opt %s -verify-diagnostics -split-input-file

// -----

// Verify: dimension size not divisible by tile_size.
func.func @bad_divisibility(%input : tensor<4x100xf32>,
    %output : tensor<4x4x32xf32>, %max : tensor<4x4xf32>, %den : tensor<4x4xf32>)
    -> (tensor<4x4x32xf32>, tensor<4x4xf32>, tensor<4x4xf32>) {
  // expected-error @+1 {{'linalg.local_softmax' op dimension size (100) must be divisible by tile_size (32)}}
  %0:3 = linalg.local_softmax dimension(1) tile_size(32)
    ins(%input : tensor<4x100xf32>)
    outs(%output : tensor<4x4x32xf32>, %max : tensor<4x4xf32>, %den : tensor<4x4xf32>)
    -> tensor<4x4x32xf32>, tensor<4x4xf32>, tensor<4x4xf32>
  return %0#0, %0#1, %0#2 : tensor<4x4x32xf32>, tensor<4x4xf32>, tensor<4x4xf32>
}

// -----

// Verify: dimension out of range.
func.func @bad_dimension(%input : tensor<4x128xf32>,
    %output : tensor<4x4x32xf32>, %max : tensor<4x4xf32>, %den : tensor<4x4xf32>)
    -> (tensor<4x4x32xf32>, tensor<4x4xf32>, tensor<4x4xf32>) {
  // expected-error @+1 {{'linalg.local_softmax' op incorrect dimension specified}}
  %0:3 = linalg.local_softmax dimension(5) tile_size(32)
    ins(%input : tensor<4x128xf32>)
    outs(%output : tensor<4x4x32xf32>, %max : tensor<4x4xf32>, %den : tensor<4x4xf32>)
    -> tensor<4x4x32xf32>, tensor<4x4xf32>, tensor<4x4xf32>
  return %0#0, %0#1, %0#2 : tensor<4x4x32xf32>, tensor<4x4xf32>, tensor<4x4xf32>
}

// -----

// Verify: output rank must be input rank + 1.
func.func @bad_output_rank(%input : tensor<4x128xf32>,
    %output : tensor<4x128xf32>, %max : tensor<4x4xf32>, %den : tensor<4x4xf32>)
    -> (tensor<4x128xf32>, tensor<4x4xf32>, tensor<4x4xf32>) {
  // expected-error @+1 {{'linalg.local_softmax' op output rank must be input rank + 1}}
  %0:3 = linalg.local_softmax dimension(1) tile_size(32)
    ins(%input : tensor<4x128xf32>)
    outs(%output : tensor<4x128xf32>, %max : tensor<4x4xf32>, %den : tensor<4x4xf32>)
    -> tensor<4x128xf32>, tensor<4x4xf32>, tensor<4x4xf32>
  return %0#0, %0#1, %0#2 : tensor<4x128xf32>, tensor<4x4xf32>, tensor<4x4xf32>
}

// -----

// Verify: max rank must equal input rank.
func.func @bad_max_rank(%input : tensor<4x128xf32>,
    %output : tensor<4x4x32xf32>, %max : tensor<4x4x32xf32>, %den : tensor<4x4xf32>)
    -> (tensor<4x4x32xf32>, tensor<4x4x32xf32>, tensor<4x4xf32>) {
  // expected-error @+1 {{'linalg.local_softmax' op max rank must equal input rank}}
  %0:3 = linalg.local_softmax dimension(1) tile_size(32)
    ins(%input : tensor<4x128xf32>)
    outs(%output : tensor<4x4x32xf32>, %max : tensor<4x4x32xf32>, %den : tensor<4x4xf32>)
    -> tensor<4x4x32xf32>, tensor<4x4x32xf32>, tensor<4x4xf32>
  return %0#0, %0#1, %0#2 : tensor<4x4x32xf32>, tensor<4x4x32xf32>, tensor<4x4xf32>
}

// -----

// Verify: tile_size must be positive.
func.func @bad_tile_size(%input : tensor<4x128xf32>,
    %output : tensor<4x4x32xf32>, %max : tensor<4x4xf32>, %den : tensor<4x4xf32>)
    -> (tensor<4x4x32xf32>, tensor<4x4xf32>, tensor<4x4xf32>) {
  // expected-error @+1 {{'linalg.local_softmax' op tile_size must be positive}}
  %0:3 = linalg.local_softmax dimension(1) tile_size(0)
    ins(%input : tensor<4x128xf32>)
    outs(%output : tensor<4x4x32xf32>, %max : tensor<4x4xf32>, %den : tensor<4x4xf32>)
    -> tensor<4x4x32xf32>, tensor<4x4xf32>, tensor<4x4xf32>
  return %0#0, %0#1, %0#2 : tensor<4x4x32xf32>, tensor<4x4xf32>, tensor<4x4xf32>
}
