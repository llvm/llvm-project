// RUN: mlir-opt -split-input-file -verify-diagnostics -tosa-narrow-i64-to-i32="convert-function-boundaries=0" %s | FileCheck %s --allow-unused-prefixes --check-prefixes=COMMON,DEFAULT
// RUN: mlir-opt -split-input-file -verify-diagnostics -tosa-narrow-i64-to-i32="convert-function-boundaries=1" %s | FileCheck %s --allow-unused-prefixes --check-prefixes=COMMON,FUNCBOUND

// -----

// CHECK-LABEL: test_i64_argmax
func.func @test_i64_argmax(%arg0: tensor<1x513x513x19xi8>) -> tensor<1x513x513xi64> {
  // COMMON: %[[ARGMAX:.*]] = tosa.argmax %arg0 {axis = 3 : i32} : (tensor<1x513x513x19xi8>) -> tensor<1x513x513xi32>
  %0 = tosa.argmax %arg0 {axis = 3 : i32} : (tensor<1x513x513x19xi8>) -> tensor<1x513x513xi64>

  // DEFAULT: %[[CAST:.*]] = tosa.cast %[[ARGMAX]] : (tensor<1x513x513xi32>) -> tensor<1x513x513xi64>
  // FUNCBOUND: return %[[ARGMAX]] : tensor<1x513x513xi32>
  return %0 : tensor<1x513x513xi64>
}

// -----

// CHECK-LABEL: test_i64_argmax_cast
func.func @test_i64_argmax_cast(%arg0: tensor<1x513x513x19xi8>) -> tensor<1x513x513xf32> {
  // COMMON: %[[ARGMAX:.*]] = tosa.argmax %arg0 {axis = 3 : i32} : (tensor<1x513x513x19xi8>) -> tensor<1x513x513xi32>
  %0 = tosa.argmax %arg0 {axis = 3 : i32} : (tensor<1x513x513x19xi8>) -> tensor<1x513x513xi64>
  // COMMON: tosa.cast %[[ARGMAX]] : (tensor<1x513x513xi32>) -> tensor<1x513x513xf32>
  %1 = tosa.cast %0 : (tensor<1x513x513xi64>) -> tensor<1x513x513xf32>
  return %1 : tensor<1x513x513xf32>
}

// -----

// CHECK-LABEL: test_i64_argmax_large_axis_dim
func.func @test_i64_argmax_large_axis_dim(%arg0: tensor<1x513x513x2147483650xi8>) -> tensor<1x513x513xi64> {
  // expected-error @+1 {{failed to legalize operation 'tosa.argmax'}}
  %0 = tosa.argmax %arg0 {axis = 3 : i32} : (tensor<1x513x513x2147483650xi8>) -> tensor<1x513x513xi64>
  return %0 : tensor<1x513x513xi64>
}

// -----

// CHECK-LABEL: test_add
func.func @test_add(%arg0: tensor<13x21x1xi64>, %arg1: tensor<13x21x3xi64>) -> tensor<13x21x3xi64> {
  // expected-error @+1 {{failed to legalize operation 'tosa.add'}}
  %0 = tosa.add %arg0, %arg1 : (tensor<13x21x1xi64>, tensor<13x21x3xi64>) -> tensor<13x21x3xi64>
  return %0 : tensor<13x21x3xi64>
}

// -----

// CHECK-LABEL: test_regions
func.func @test_regions(%arg0: tensor<1x2xi32>, %arg1: tensor<1xi32>, %arg2: tensor<i1>) -> tensor<1xi32> {
  // COMMON: %[[IF_RESULT:.*]] = tosa.cond_if %arg2 : tensor<i1> -> tensor<1xi32>
  %0 = tosa.cond_if %arg2 : tensor<i1> -> tensor<1xi32> {
    // COMMON: %[[ARGMAX:.*]] = tosa.argmax %arg0 {axis = 1 : i32} : (tensor<1x2xi32>) -> tensor<1xi32>
    %1 = tosa.argmax %arg0 {axis = 1 : i32} : (tensor<1x2xi32>) -> tensor<1xi64>
    // COMMON: %[[CAST:.*]] = tosa.cast %[[ARGMAX]] : (tensor<1xi32>) -> tensor<1xi32>
    %2 = tosa.cast %1 : (tensor<1xi64>) -> tensor<1xi32>
    // COMMON: tosa.yield %[[CAST]] : tensor<1xi32>
    tosa.yield %2 : tensor<1xi32>
  } else {
    tosa.yield %arg1 : tensor<1xi32>
  }
  // COMMON: return %[[IF_RESULT]] : tensor<1xi32>
  return %0 : tensor<1xi32>
}

// -----

// CHECK-LABEL: test_concat
func.func @test_concat(%arg0: tensor<13x21x3xi64>, %arg1: tensor<13x21x3xi64>) -> tensor<26x21x3xi64> {
  // COMMON: tosa.concat %{{.*}}, %{{.*}} {axis = 0 : i32} : (tensor<13x21x3xi32>, tensor<13x21x3xi32>) -> tensor<26x21x3xi32>
  %0 = tosa.concat %arg0, %arg1 {axis = 0 : i32} : (tensor<13x21x3xi64>, tensor<13x21x3xi64>) -> tensor<26x21x3xi64>
  return %0 : tensor<26x21x3xi64>
}

// -----

// CHECK-LABEL: test_pad
func.func @test_pad(%arg0: tensor<13x21x3xi64>, %arg1: tensor<1xi64>) -> tensor<15x23x5xi64> {
  %padding = tosa.const_shape {values = dense<1> : tensor<6xindex>} : () -> !tosa.shape<6>
  // COMMON: tosa.pad %{{.*}}, %{{.*}}, %{{.*}} : (tensor<13x21x3xi32>, !tosa.shape<6>, tensor<1xi32>) -> tensor<15x23x5xi32>
  %1 = tosa.pad %arg0, %padding, %arg1 : (tensor<13x21x3xi64>, !tosa.shape<6>, tensor<1xi64>) -> tensor<15x23x5xi64>
  return %1 : tensor<15x23x5xi64>
}

// -----

// CHECK-LABEL: test_reshape
func.func @test_reshape(%arg0: tensor<13x21x3xi64>) -> tensor<1x819xi64> {
  %1 = tosa.const_shape {values = dense<[1, 819]> : tensor<2xindex>} : () -> !tosa.shape<2>
  // COMMON: tosa.reshape %{{.*}}, %{{.*}} : (tensor<13x21x3xi32>, !tosa.shape<2>) -> tensor<1x819xi32>
  %0 = tosa.reshape %arg0, %1 : (tensor<13x21x3xi64>, !tosa.shape<2>) -> tensor<1x819xi64>
  return %0 : tensor<1x819xi64>
}

// -----

// CHECK-LABEL: test_reverse
func.func @test_reverse(%arg0: tensor<13x21x3xi64>) -> tensor<13x21x3xi64> {
  // COMMON: tosa.reverse %{{.*}} {axis = 0 : i32} : (tensor<13x21x3xi32>) -> tensor<13x21x3xi32>
  %0 = tosa.reverse %arg0 {axis = 0 : i32} : (tensor<13x21x3xi64>) -> tensor<13x21x3xi64>
  return %0 : tensor<13x21x3xi64>
}

// -----

// CHECK-LABEL: test_slice
func.func @test_slice(%arg0: tensor<13x21x3xi64>) -> tensor<4x11x1xi64> {
  %0 = tosa.const_shape {values = dense<[4, 11, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %1 = tosa.const_shape {values = dense<[6, 8, 0]> : tensor<3xindex>} : () -> !tosa.shape<3>
  // COMMON: tosa.slice %{{.*}}, %{{.*}}, %{{.*}} : (tensor<13x21x3xi32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<4x11x1xi32>
  %2 = tosa.slice %arg0, %0, %1 : (tensor<13x21x3xi64>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<4x11x1xi64>
  return %2 : tensor<4x11x1xi64>
}

// -----

// CHECK-LABEL: test_tile
func.func @test_tile(%arg0: tensor<13x21x3xi64>) -> tensor<39x21x6xi64> {
  %cst = tosa.const_shape { values = dense<[3, 1, 2]> : tensor<3xindex> } : () -> !tosa.shape<3>
  // COMMON: tosa.tile %{{.*}}, %{{.*}} : (tensor<13x21x3xi32>, !tosa.shape<3>) -> tensor<39x21x6xi32>
  %0 = tosa.tile %arg0, %cst: (tensor<13x21x3xi64>, !tosa.shape<3>) -> tensor<39x21x6xi64>
  return %0 : tensor<39x21x6xi64>
}

// -----

// CHECK-LABEL: transpose
func.func @test_transpose(%arg0: tensor<13x21x3xi64>) -> tensor<3x13x21xi64> {
  // COMMON: tosa.transpose %{{.*}} {perms = array<i32: 2, 0, 1>} : (tensor<13x21x3xi32>) -> tensor<3x13x21xi32>
  %1 = tosa.transpose %arg0 {perms = array<i32: 2, 0, 1>} : (tensor<13x21x3xi64>) -> tensor<3x13x21xi64>
  return %1 : tensor<3x13x21xi64>
}

// -----

// CHECK-LABEL: test_transition_to_i64
func.func @test_transition_to_i64(%arg0: tensor<1xi32>) -> tensor<1xi64> {
  // COMMON: %[[CAST:.*]] = tosa.cast %arg0 : (tensor<1xi32>) -> tensor<1xi32>
  %0 = tosa.cast %arg0 : (tensor<1xi32>) -> tensor<1xi64>
  // COMMON: %[[IDENTITY1:.*]] = tosa.identity %[[CAST]] : (tensor<1xi32>) -> tensor<1xi32>
  %1 = tosa.identity %0 : (tensor<1xi64>) -> tensor<1xi64>
  // COMMON: %[[IDENTITY2:.*]] = tosa.identity %[[IDENTITY1]] : (tensor<1xi32>) -> tensor<1xi32>
  %2 = tosa.identity %1 : (tensor<1xi64>) -> tensor<1xi64>
  // DEFAULT: %[[OUT_CAST:.*]] = tosa.cast %[[IDENTITY2]] : (tensor<1xi32>) -> tensor<1xi64>
  // DEFAULT: return %[[OUT_CAST]] : tensor<1xi64>
  // FUNCBOUND: return %[[IDENTITY2]] : tensor<1xi32>
  return %2 : tensor<1xi64>
}

// -----

// CHECK-LABEL: test_transition_from_i64
func.func @test_transition_from_i64(%arg0: tensor<1xi64>) -> tensor<1xi32> {
  // DEFAULT: %[[CAST:.*]] = tosa.cast %arg0 : (tensor<1xi64>) -> tensor<1xi32>
  // DEFAULT: %[[IDENTITY1:.*]] = tosa.identity %[[CAST]] : (tensor<1xi32>) -> tensor<1xi32>
  // FUNCBOUND: %[[IDENTITY1:.*]] = tosa.identity %arg0 : (tensor<1xi32>) -> tensor<1xi32>
  %0 = tosa.identity %arg0 : (tensor<1xi64>) -> tensor<1xi64>
  // COMMON: %[[IDENTITY2:.*]] = tosa.identity %[[IDENTITY1]] : (tensor<1xi32>) -> tensor<1xi32>
  %1 = tosa.identity %0 : (tensor<1xi64>) -> tensor<1xi64>
  // COMMON: %[[OUT_CAST:.*]] = tosa.cast %[[IDENTITY2]] : (tensor<1xi32>) -> tensor<1xi32>
  %2 = tosa.cast %1 : (tensor<1xi64>) -> tensor<1xi32>
  // COMMON: return %[[OUT_CAST]] : tensor<1xi32>
  return %2 : tensor<1xi32>
}
