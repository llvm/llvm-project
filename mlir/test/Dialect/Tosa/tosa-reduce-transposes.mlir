// RUN: mlir-opt --verify-diagnostics --split-input-file --verify-each --tosa-reduce-transposes %s | FileCheck %s

// CHECK-LABEL: @test_transpose_tracks_to_nullifying_single_step
// CHECK-NEXT: %[[RESULT:.*]] = tosa.ceil %arg0
// CHECK-NEXT: return %[[RESULT]]
func.func @test_transpose_tracks_to_nullifying_single_step(%arg0: tensor<1x2x3x4xi32>) -> tensor<1x2x3x4xi32> {
  %perms0 = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %0 = tosa.transpose %arg0, %perms0 : (tensor<1x2x3x4xi32>, tensor<4xi32>) -> tensor<1x3x4x2xi32>
  %ceil = tosa.ceil %0 : (tensor<1x3x4x2xi32>) -> tensor<1x3x4x2xi32>
  %perms1 = "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  %1 = tosa.transpose %ceil, %perms1 : (tensor<1x3x4x2xi32>, tensor<4xi32>) -> tensor<1x2x3x4xi32>
  return %1 : tensor<1x2x3x4xi32>
}

// -----

// CHECK-LABEL: @test_transpose_tracks_to_nullifying_multi_unary_step
// CHECK-NEXT: %[[CLAMP:.*]] = tosa.clamp %arg0
// CHECK-NEXT: %[[ABS:.*]] = tosa.abs %[[CLAMP]]
// CHECK-NEXT: %[[NOT:.*]] = tosa.bitwise_not %[[ABS]]
// CHECK-NEXT: return %[[NOT]]
func.func @test_transpose_tracks_to_nullifying_multi_unary_step(%arg0: tensor<1x2x3x4xi32>) -> tensor<1x2x3x4xi32> {
  %perms0 = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %0 = tosa.transpose %arg0, %perms0 : (tensor<1x2x3x4xi32>, tensor<4xi32>) -> tensor<1x3x4x2xi32>
  %clamp = tosa.clamp %0 {max_val = 1 : i32, min_val = 0 : i32} : (tensor<1x3x4x2xi32>) -> tensor<1x3x4x2xi32>
  %abs = tosa.abs %clamp : (tensor<1x3x4x2xi32>) -> tensor<1x3x4x2xi32>
  %bitwise_not = tosa.bitwise_not %abs : (tensor<1x3x4x2xi32>) -> tensor<1x3x4x2xi32>
  %perms1 = "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  %1 = tosa.transpose %bitwise_not, %perms1 : (tensor<1x3x4x2xi32>, tensor<4xi32>) -> tensor<1x2x3x4xi32>
  return %1 : tensor<1x2x3x4xi32>
}

// -----

// CHECK-LABEL: @test_transpose_tracks_to_nullifying_diverging_binary
// CHECK-NEXT: %[[CLAMP:.*]] = tosa.clamp %arg0
// CHECK-NEXT: %[[ABS:.*]] = tosa.abs %arg1
// CHECK-NEXT: %[[ADD:.*]] = tosa.add %[[CLAMP]], %[[ABS]]
// CHECK-NEXT: return %[[ADD]]
func.func @test_transpose_tracks_to_nullifying_diverging_binary(%arg0: tensor<1x2x3x4xi32>, %arg1: tensor<1x2x3x4xi32>) -> tensor<1x2x3x4xi32> {
  %perms0 = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %transpose0 = tosa.transpose %arg0, %perms0 : (tensor<1x2x3x4xi32>, tensor<4xi32>) -> tensor<1x3x4x2xi32>
  %transpose1 = tosa.transpose %arg1, %perms0 : (tensor<1x2x3x4xi32>, tensor<4xi32>) -> tensor<1x3x4x2xi32>
  %clamp = tosa.clamp %transpose0 {max_val = 1 : i32, min_val = 0 : i32} : (tensor<1x3x4x2xi32>) -> tensor<1x3x4x2xi32>
  %abs = tosa.abs %transpose1 : (tensor<1x3x4x2xi32>) -> tensor<1x3x4x2xi32>
  %add = tosa.add %clamp, %abs : (tensor<1x3x4x2xi32>, tensor<1x3x4x2xi32>) -> tensor<1x3x4x2xi32>
  %perms1 = "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  %result = tosa.transpose %add, %perms1 : (tensor<1x3x4x2xi32>, tensor<4xi32>) -> tensor<1x2x3x4xi32>
  return %result : tensor<1x2x3x4xi32>
}

// -----


// CHECK-LABEL: @test_transpose_tracks_to_nullifying_diverging_binary_with_broadcasting
// CHECK-NEXT: %[[CLAMP:.*]] = tosa.clamp %arg0
// CHECK-NEXT: %[[ABS:.*]] = tosa.abs %arg1
// CHECK-NEXT: %[[ADD:.*]] = tosa.add %[[CLAMP]], %[[ABS]]
// CHECK-NEXT: return %[[ADD]]
func.func @test_transpose_tracks_to_nullifying_diverging_binary_with_broadcasting(%arg0: tensor<1x2x3x4xi32>, %arg1: tensor<1x2x1x4xi32>) -> tensor<1x2x3x4xi32> {
  %perms0 = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %transpose0 = tosa.transpose %arg0, %perms0 : (tensor<1x2x3x4xi32>, tensor<4xi32>) -> tensor<1x3x4x2xi32>
  %transpose1 = tosa.transpose %arg1, %perms0 : (tensor<1x2x1x4xi32>, tensor<4xi32>) -> tensor<1x1x4x2xi32>
  %clamp = tosa.clamp %transpose0 {max_val = 1 : i32, min_val = 0 : i32} : (tensor<1x3x4x2xi32>) -> tensor<1x3x4x2xi32>
  %abs = tosa.abs %transpose1 : (tensor<1x1x4x2xi32>) -> tensor<1x1x4x2xi32>
  %add = tosa.add %clamp, %abs : (tensor<1x3x4x2xi32>, tensor<1x1x4x2xi32>) -> tensor<1x3x4x2xi32>
  %perms1 = "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  %result = tosa.transpose %add, %perms1 : (tensor<1x3x4x2xi32>, tensor<4xi32>) -> tensor<1x2x3x4xi32>
  return %result : tensor<1x2x3x4xi32>
}

// -----

// CHECK-LABEL: @test_transpose_tracks_to_nullifying__converging_binary
// CHECK-NEXT: %[[RESULT:.*]] = tosa.add %arg0, %arg0
// CHECK-NEXT: return %[[RESULT]]
func.func @test_transpose_tracks_to_nullifying__converging_binary(%arg0: tensor<1x2x3x4xi32>) -> tensor<1x2x3x4xi32> {
  %perms0 = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %0 = tosa.transpose %arg0, %perms0 : (tensor<1x2x3x4xi32>, tensor<4xi32>) -> tensor<1x3x4x2xi32>
  %clamp = tosa.add %0, %0 : (tensor<1x3x4x2xi32>, tensor<1x3x4x2xi32>) -> tensor<1x3x4x2xi32>
  %perms1 = "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  %1 = tosa.transpose %clamp, %perms1 : (tensor<1x3x4x2xi32>, tensor<4xi32>) -> tensor<1x2x3x4xi32>
  return %1 : tensor<1x2x3x4xi32>
}

// -----

// CHECK-LABEL: @test_torch_conv2d_with_elementwise_in_between
// CHECK: %[[CONV1:.*]] = tosa.conv2d
// CHECK: %[[CEIL:.*]] = tosa.ceil %[[CONV1]]
// CHECK: %[[CONV2:.*]] = tosa.conv2d %[[CEIL]]
// CHECK: %[[FLOOR:.*]] = tosa.floor %[[CONV2]]
// CHECK: %[[CONV3:.*]] = tosa.conv2d %[[FLOOR]]
// CHECK: %[[RES:.*]] = tosa.transpose %[[CONV3]]
// CHECK: return %[[RES]]
func.func @test_torch_conv2d_with_elementwise_in_between(%arg0: tensor<3x3x10x10xf32>) -> tensor<3x3x7x7xf32> {
    %0 = "tosa.const"() <{value = dense_resource<torch_tensor_3_torch.float32_2> : tensor<3xf32>}> : () -> tensor<3xf32>
    %1 = "tosa.const"() <{value = dense_resource<torch_tensor_3_3_2_2_torch.float32_2> : tensor<3x3x2x2xf32>}> : () -> tensor<3x3x2x2xf32>
    %2 = "tosa.const"() <{value = dense_resource<torch_tensor_3_torch.float32_1> : tensor<3xf32>}> : () -> tensor<3xf32>
    %3 = "tosa.const"() <{value = dense_resource<torch_tensor_3_3_2_2_torch.float32_1> : tensor<3x3x2x2xf32>}> : () -> tensor<3x3x2x2xf32>
    %4 = "tosa.const"() <{value = dense_resource<torch_tensor_3_3_2_2_torch.float32> : tensor<3x3x2x2xf32>}> : () -> tensor<3x3x2x2xf32>
    %5 = "tosa.const"() <{value = dense_resource<torch_tensor_3_torch.float32> : tensor<3xf32>}> : () -> tensor<3xf32>
    %6 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %7 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %8 = tosa.transpose %arg0, %6 : (tensor<3x3x10x10xf32>, tensor<4xi32>) -> tensor<3x10x10x3xf32>
    %9 = tosa.transpose %4, %6 : (tensor<3x3x2x2xf32>, tensor<4xi32>) -> tensor<3x2x2x3xf32>
    %10 = tosa.conv2d %8, %9, %5 {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<3x10x10x3xf32>, tensor<3x2x2x3xf32>, tensor<3xf32>) -> tensor<3x9x9x3xf32>
    %11 = tosa.transpose %10, %7 : (tensor<3x9x9x3xf32>, tensor<4xi32>) -> tensor<3x3x9x9xf32>
    %12 = tosa.ceil %11 : (tensor<3x3x9x9xf32>) -> tensor<3x3x9x9xf32>
    %13 = tosa.transpose %12, %6 : (tensor<3x3x9x9xf32>, tensor<4xi32>) -> tensor<3x9x9x3xf32>
    %14 = tosa.transpose %3, %6 : (tensor<3x3x2x2xf32>, tensor<4xi32>) -> tensor<3x2x2x3xf32>
    %15 = tosa.conv2d %13, %14, %2 {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<3x9x9x3xf32>, tensor<3x2x2x3xf32>, tensor<3xf32>) -> tensor<3x8x8x3xf32>
    %16 = tosa.transpose %15, %7 : (tensor<3x8x8x3xf32>, tensor<4xi32>) -> tensor<3x3x8x8xf32>
    %17 = tosa.floor %16 : (tensor<3x3x8x8xf32>) -> tensor<3x3x8x8xf32>
    %18 = tosa.transpose %17, %6 : (tensor<3x3x8x8xf32>, tensor<4xi32>) -> tensor<3x8x8x3xf32>
    %19 = tosa.transpose %1, %6 : (tensor<3x3x2x2xf32>, tensor<4xi32>) -> tensor<3x2x2x3xf32>
    %20 = tosa.conv2d %18, %19, %0 {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<3x8x8x3xf32>, tensor<3x2x2x3xf32>, tensor<3xf32>) -> tensor<3x7x7x3xf32>
    %21 = tosa.transpose %20, %7 : (tensor<3x7x7x3xf32>, tensor<4xi32>) -> tensor<3x3x7x7xf32>
    return %21 : tensor<3x3x7x7xf32>
}

// -----

// CHECK-LABEL: @test_mulop_conversion
// CHECK-NEXT: %[[SHIFT:.*]] = "tosa.const"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK-NEXT: %[[RES:.*]] = tosa.mul %arg0, %arg1, %[[SHIFT]]
// CHECK-NEXT: return %[[RES]]
func.func @test_mulop_conversion(%arg0: tensor<1x2x3x4xi32>, %arg1: tensor<1x2x3x4xi32>) -> tensor<1x2x3x4xi32> {
  %perms0 = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %transpose0 = tosa.transpose %arg0, %perms0 : (tensor<1x2x3x4xi32>, tensor<4xi32>) -> tensor<1x3x4x2xi32>
  %transpose1 = tosa.transpose %arg1, %perms0 : (tensor<1x2x3x4xi32>, tensor<4xi32>) -> tensor<1x3x4x2xi32>
  %shift = "tosa.const"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %mul = tosa.mul %transpose0, %transpose1, %shift : (tensor<1x3x4x2xi32>, tensor<1x3x4x2xi32>, tensor<1xi8>) -> tensor<1x3x4x2xi32>
  %perms1 = "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  %result = tosa.transpose %mul, %perms1 : (tensor<1x3x4x2xi32>, tensor<4xi32>) -> tensor<1x2x3x4xi32>
  return %result : tensor<1x2x3x4xi32>
}

// -----

// COM: this case is a reshape we don't convert, since can't fold the transpose into it.
// COM: a transform actually occurs underneath the hood, but it results in identical IR.
// CHECK-LABEL: @test_basic_non_broadcasting_reshape
// CHECK: "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK: tosa.reshape %arg0 {new_shape = array<i64: 1, 3, 2>} : (tensor<2x3xi32>) -> tensor<1x3x2xi32>
// CHECK: tosa.transpose %1, %0 : (tensor<1x3x2xi32>, tensor<3xi32>) -> tensor<1x2x3xi32>
func.func @test_basic_non_broadcasting_reshape(%arg0: tensor<2x3xi32>) -> tensor<1x2x3xi32> {
  %perms = "tosa.const"() {value = dense<[0, 2, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
  %1 = tosa.reshape %arg0 {new_shape = array<i64: 1, 3, 2>} : (tensor<2x3xi32>) -> tensor<1x3x2xi32>
  %2 = tosa.transpose %1, %perms : (tensor<1x3x2xi32>, tensor<3xi32>) -> tensor<1x2x3xi32>
  return %2 : tensor<1x2x3xi32>
}

// -----

// CHECK-LABEL: @test_dynamic_broadcasting_reshape
// CHECK: %[[RES:.*]] = tosa.reshape %arg0 {new_shape = array<i64: 1, 1, -1>} : (tensor<?xi32>) -> tensor<1x1x?xi32>
// CHECK: return %[[RES]]
func.func @test_dynamic_broadcasting_reshape(%arg0: tensor<?xi32>) -> tensor<1x1x?xi32> {
  %perms = "tosa.const"() {value = dense<[0, 2, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
  %1 = tosa.reshape %arg0 {new_shape = array<i64: 1, -1, 1>} : (tensor<?xi32>) -> tensor<1x?x1xi32>
  %2 = tosa.transpose %1, %perms : (tensor<1x?x1xi32>, tensor<3xi32>) -> tensor<1x1x?xi32>
  return %2 : tensor<1x1x?xi32>
}

// -----

// CHECK-LABEL: @test_reshape_for_broadcast
// CHECK-DAG: %[[RESHAPE_INPUT:.*]] = "tosa.const"() <{value = dense<[1, 2, 3, 4]>
// CHECK-DAG: %[[RESHAPE:.*]] = tosa.reshape %[[RESHAPE_INPUT]] {new_shape = array<i64: 4, 1, 1>}
// CHECK-DAG: %[[ADD:.*]] = tosa.add %arg0, %[[RESHAPE]]
// CHECK: return %[[ADD]]
func.func @test_reshape_for_broadcast(%arg0: tensor<4x3x2xi32>) -> tensor<4x3x2xi32> {
  %0 = "tosa.const"() {value = dense<[1,2,3,4]> : tensor<4xi32>} : () -> tensor<4xi32>
  %reshape = tosa.reshape %0 {new_shape = array<i64: 1, 1, 4>} : (tensor<4xi32>) -> tensor<1x1x4xi32>
  %perms0 = "tosa.const"() {value = dense<[2, 1, 0]> : tensor<3xi32>} : () -> tensor<3xi32>
  %transpose0 = tosa.transpose %arg0, %perms0 : (tensor<4x3x2xi32>, tensor<3xi32>) -> tensor<2x3x4xi32>
  %add = tosa.add %transpose0, %reshape : (tensor<2x3x4xi32>, tensor<1x1x4xi32>) -> tensor<2x3x4xi32>
  %transpose1 = tosa.transpose %add, %perms0 : (tensor<2x3x4xi32>, tensor<3xi32>) -> tensor<4x3x2xi32>
  return %transpose1 : tensor<4x3x2xi32>
}

// -----

// COM: taken directly from ResNet18 translation.
// COM: changes: %74 as argument instead of result of conv2d

// CHECK-LABEL: @test_resnet18_common_case
// COM: note that %74 is now represented by %arg2
// CHECK-DAG: %[[VAL_2:.*]] = "tosa.const"() <{value = dense_resource<torch_tensor_64_torch.float32_1> : tensor<64xf32>}> : () -> tensor<64xf32>
// CHECK-DAG: %[[VAL_3:.*]] = "tosa.const"() <{value = dense_resource<torch_tensor_64_torch.float32> : tensor<64xf32>}> : () -> tensor<64xf32>
// CHECK-DAG: %[[VAL_4:.*]] = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK-DAG: %[[VAL_5:.*]] = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK-DAG: %[[VAL_6:.*]] = tosa.add %arg1, %[[VAL_4]] : (tensor<64xf32>, tensor<1xf32>) -> tensor<64xf32>
// CHECK-DAG: %[[VAL_7:.*]] = tosa.pow %[[VAL_6]], %[[VAL_5]] : (tensor<64xf32>, tensor<1xf32>) -> tensor<64xf32>
// CHECK-DAG: %[[VAL_8:.*]] = tosa.reciprocal %[[VAL_7]] : (tensor<64xf32>) -> tensor<64xf32>
// CHECK-DAG: %[[VAL_9:.*]] = tosa.reshape %arg0 {new_shape = array<i64: 1, 1, 1, 64>} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
// CHECK-DAG: %[[VAL_10:.*]] = tosa.sub %arg2, %[[VAL_9]] : (tensor<1x112x112x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x112x112x64xf32>
// CHECK-DAG: %[[VAL_11:.*]] = tosa.reshape %[[VAL_8]] {new_shape = array<i64: 1, 1, 1, 64>} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
// CHECK-DAG: %[[VAL_12:.*]] = tosa.mul %[[VAL_10]], %[[VAL_11]] : (tensor<1x112x112x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x112x112x64xf32>
// CHECK-DAG: %[[VAL_13:.*]] = tosa.reshape %[[VAL_3]] {new_shape = array<i64: 1, 1, 1, 64>} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
// CHECK-DAG: %[[VAL_14:.*]] = tosa.mul %[[VAL_12]], %[[VAL_13]] : (tensor<1x112x112x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x112x112x64xf32>
// CHECK-DAG: %[[VAL_15:.*]] = tosa.reshape %[[VAL_2]] {new_shape = array<i64: 1, 1, 1, 64>} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
// CHECK-DAG: %[[VAL_16:.*]] = tosa.add %[[VAL_14]], %[[VAL_15]] : (tensor<1x112x112x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x112x112x64xf32>
// CHECK-DAG: %[[VAL_17:.*]] = tosa.clamp %[[VAL_16]] {max_val = 3.40282347E+38 : f32, min_val = 0.000000e+00 : f32} : (tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32>
// CHECK: return %[[VAL_17]] : tensor<1x112x112x64xf32>

func.func @test_resnet18_common_case(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %74: tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32> {
    %59 = "tosa.const"() <{value = dense_resource<torch_tensor_64_torch.float32_1> : tensor<64xf32>}> : () -> tensor<64xf32>
    %60 = "tosa.const"() <{value = dense_resource<torch_tensor_64_torch.float32> : tensor<64xf32>}> : () -> tensor<64xf32>
    %63 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %64 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %69 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1xf32>}> : () -> tensor<1xf32>
    %70 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1xf32>}> : () -> tensor<1xf32>
    %75 = tosa.transpose %74, %64 : (tensor<1x112x112x64xf32>, tensor<4xi32>) -> tensor<1x64x112x112xf32>
    %76 = tosa.add %arg1, %69 : (tensor<64xf32>, tensor<1xf32>) -> tensor<64xf32>
    %77 = tosa.pow %76, %70 : (tensor<64xf32>, tensor<1xf32>) -> tensor<64xf32>
    %78 = tosa.reciprocal %77 : (tensor<64xf32>) -> tensor<64xf32>
    %79 = tosa.reshape %arg0 {new_shape = array<i64: 1, 64, 1, 1>} : (tensor<64xf32>) -> tensor<1x64x1x1xf32>
    %80 = tosa.sub %75, %79 : (tensor<1x64x112x112xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x112x112xf32>
    %81 = tosa.reshape %78 {new_shape = array<i64: 1, 64, 1, 1>} : (tensor<64xf32>) -> tensor<1x64x1x1xf32>
    %82 = tosa.mul %80, %81 : (tensor<1x64x112x112xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x112x112xf32>
    %83 = tosa.reshape %60 {new_shape = array<i64: 1, 64, 1, 1>} : (tensor<64xf32>) -> tensor<1x64x1x1xf32>
    %84 = tosa.mul %82, %83 : (tensor<1x64x112x112xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x112x112xf32>
    %85 = tosa.reshape %59 {new_shape = array<i64: 1, 64, 1, 1>} : (tensor<64xf32>) -> tensor<1x64x1x1xf32>
    %86 = tosa.add %84, %85 : (tensor<1x64x112x112xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x112x112xf32>
    %87 = tosa.clamp %86 {max_val = 3.40282347E+38 : f32, min_val = 0.000000e+00 : f32} : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %88 = tosa.transpose %87, %63 : (tensor<1x64x112x112xf32>, tensor<4xi32>) -> tensor<1x112x112x64xf32>
    return %88 : tensor<1x112x112x64xf32>
}


// -----

// CHECK-LABEL: @test_back_to_back_nullifiers
// CHECK: %[[PERMS:.*]] = "tosa.const"
// CHECK: %[[RES:.*]] = tosa.transpose %arg0, %[[PERMS]]
// CHECK: return %[[RES]]
func.func @test_back_to_back_nullifiers(%arg0: tensor<2x3xi32>) -> tensor<3x2xi32> {
  %perms = "tosa.const"() {value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
  %0 = tosa.transpose %arg0, %perms : (tensor<2x3xi32>, tensor<2xi32>) -> tensor<3x2xi32>
  %1 = tosa.transpose %0, %perms : (tensor<3x2xi32>, tensor<2xi32>) -> tensor<2x3xi32>
  %2 = tosa.transpose %1, %perms : (tensor<2x3xi32>, tensor<2xi32>) -> tensor<3x2xi32>
  return %2 : tensor<3x2xi32>
}

// -----

// CHECK-LABEL: @test_back_to_back_nullifiers_different_transposes
// CHECK: %[[PERMS:.*]] = "tosa.const"
// CHECK: %[[RES:.*]] = tosa.transpose %arg0, %[[PERMS]]
// CHECK: return %[[RES]]
func.func @test_back_to_back_nullifiers_different_transposes(%arg0: tensor<2x3x4x5xi32>) -> tensor<2x4x5x3xi32> {
  %perms0 = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %perms1 = "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  %0 = tosa.transpose %arg0, %perms0 : (tensor<2x3x4x5xi32>, tensor<4xi32>) -> tensor<2x4x5x3xi32>
  %1 = tosa.transpose %0, %perms1 : (tensor<2x4x5x3xi32>, tensor<4xi32>) -> tensor<2x3x4x5xi32>
  %2 = tosa.transpose %1, %perms0 : (tensor<2x3x4x5xi32>, tensor<4xi32>) -> tensor<2x4x5x3xi32>
  return %2 : tensor<2x4x5x3xi32>
}

// -----

// CHECK-LABEL: @test_no_transform_if_outside_fan_in_cone
// CHECK: tosa.const
// CHECK: %[[CLAMP_IN:.*]] = tosa.transpose
// CHECK: %[[RES2:.*]] = tosa.clamp %[[CLAMP_IN]]
// CHECK: tosa.const
// CHECK: %[[RES1:.*]] = tosa.transpose
// CHECK: return %[[RES1]], %[[RES2]]
func.func @test_no_transform_if_outside_fan_in_cone(%arg0: tensor<3x3x3x3xi32>) -> (tensor<3x3x3x3xi32>, tensor<3x3x3x3xi32>) {
  %perms0 = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %0 = tosa.transpose %arg0, %perms0 : (tensor<3x3x3x3xi32>, tensor<4xi32>) -> tensor<3x3x3x3xi32>
  %clamp = tosa.clamp %0  {max_val = 2147483647 : i32, min_val = 0 : i32} : (tensor<3x3x3x3xi32>) -> tensor<3x3x3x3xi32>
  %perms1 = "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  %1 = tosa.transpose %clamp, %perms1 : (tensor<3x3x3x3xi32>, tensor<4xi32>) -> tensor<3x3x3x3xi32>
  return %1, %clamp : tensor<3x3x3x3xi32>, tensor<3x3x3x3xi32>
}

// -----

// CHECK-LABEL: @test_two_different_downstream_converge_to_reshape_same_perms
// CHECK-DAG: %[[RESHAPE:.*]] = tosa.reshape %arg0
// CHECK-DAG: %[[CLAMP:.*]] = tosa.clamp %[[RESHAPE]]
// CHECK: return %[[RESHAPE]], %[[CLAMP]]
func.func @test_two_different_downstream_converge_to_reshape_same_perms(%arg0: tensor<64xf32>) -> (tensor<1x1x64xf32>, tensor<1x1x64xf32>) {
  %0 = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
  %1 = tosa.reshape %arg0 {new_shape = array<i64: 1, 64, 1>} : (tensor<64xf32>) -> tensor<1x64x1xf32>
  %2 = tosa.clamp %1 {max_val = 3.40282347E+38 : f32, min_val = 0.000000e+00 : f32} : (tensor<1x64x1xf32>) -> tensor<1x64x1xf32>
  %3 = tosa.transpose %1, %0 : (tensor<1x64x1xf32>, tensor<3xi32>) -> tensor<1x1x64xf32>
  %4 = tosa.transpose %2, %0 : (tensor<1x64x1xf32>, tensor<3xi32>) -> tensor<1x1x64xf32>
  return %3, %4 : tensor<1x1x64xf32>, tensor<1x1x64xf32>
}

// -----

// CHECK-LABEL: @test_two_different_downstream_converge_to_reshape_different_perms
// CHECK-DAG: tosa.const
// CHECK-DAG: tosa.const
// CHECK-DAG: %[[RESHAPE:.*]] = tosa.reshape
// CHECK-DAG: %[[CLAMP:.*]] = tosa.clamp %[[RESHAPE]]
// CHECK-DAG: %[[RET1:.*]] = tosa.transpose
// CHECK-DAG: %[[RET2:.*]] = tosa.transpose
// CHECK-DAG: return %[[RET1]], %[[RET2]]
func.func @test_two_different_downstream_converge_to_reshape_different_perms(%arg0: tensor<64xf32>) -> (tensor<1x1x64xf32>, tensor<64x1x1xf32>) {
  %0 = "tosa.const"() <{value = dense<[1, 2, 0]> : tensor<3xi32>}> : () -> tensor<3xi32>
  %1 = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
  %2 = tosa.reshape %arg0 {new_shape = array<i64: 1, 64, 1>} : (tensor<64xf32>) -> tensor<1x64x1xf32>
  %3 = tosa.clamp %2 {max_val = 3.40282347E+38 : f32, min_val = 0.000000e+00 : f32} : (tensor<1x64x1xf32>) -> tensor<1x64x1xf32>
  %4 = tosa.transpose %2, %1 : (tensor<1x64x1xf32>, tensor<3xi32>) -> tensor<1x1x64xf32>
  %5 = tosa.transpose %3, %0 : (tensor<1x64x1xf32>, tensor<3xi32>) -> tensor<64x1x1xf32>
  return %4, %5 : tensor<1x1x64xf32>, tensor<64x1x1xf32>
}

// -----

// COM: no transform
// CHECK-LABEL: @test_outside_perms_usage_of_fan_in
// CHECK: tosa.const
// CHECK: tosa.transpose
// CHECK: tosa.clamp
// CHECK: %[[RES1:.*]] = tosa.transpose
// CHECK: %[[RES2:.*]] = tosa.add
// CHECK: return %[[RES1]], %[[RES2]]
func.func @test_outside_perms_usage_of_fan_in(%arg0: tensor<2x3xf32>, %arg1: tensor<3x2xf32>) -> (tensor<2x3xf32>, tensor<3x2xf32>) {        %0 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
  %1 = tosa.transpose %arg0, %0 : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>
  %2 = tosa.clamp %1  {max_val = 3.40282347E+38 : f32, min_val = 0.000000e+00 : f32} : (tensor<3x2xf32>) -> tensor<3x2xf32>
  %3 = tosa.transpose %2, %0 : (tensor<3x2xf32>, tensor<2xi32>) -> tensor<2x3xf32>
  %4 = tosa.add %arg1, %2 : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>
  return %3, %4: tensor<2x3xf32>, tensor<3x2xf32>
}

// -----

// COM: this use-case is important for ResNet. we want to allow these, but disallow if falls into an illegal area (outside fan-ins that get converted),
// COM: since then we would get duplicate ops.
// CHECK-LABEL: @test_use_present_in_another_valid_perms_fan_in
// CHECK-DAG: %[[NEW_CLAMP:.*]] = tosa.clamp %arg0
// CHECK-DAG: %[[NEW_ADD:.*]] = tosa.add %arg1, %[[NEW_CLAMP]]
// CHECK: return %[[NEW_CLAMP]], %[[NEW_ADD]]
func.func @test_use_present_in_another_valid_perms_fan_in(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> (tensor<2x3xf32>, tensor<2x3xf32>) {
  %0 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
  %1 = tosa.transpose %arg0, %0 : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>
  %2 = tosa.clamp %1  {max_val = 3.40282347E+38 : f32, min_val = 0.000000e+00 : f32} : (tensor<3x2xf32>) -> tensor<3x2xf32>
  %3 = tosa.transpose %2, %0 : (tensor<3x2xf32>, tensor<2xi32>) -> tensor<2x3xf32>
  %4 = tosa.transpose %arg1, %0 : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>
  %5 = tosa.add %4, %2 : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>
  %6 = tosa.transpose %5, %0 : (tensor<3x2xf32>, tensor<2xi32>) -> tensor<2x3xf32>
  return %3, %6: tensor<2x3xf32>, tensor<2x3xf32>
}

// -----

// COM: no transform, since we would get duplicates
// CHECK-LABEL: @test_two_same_perms_fan_in_but_one_doesnt_convert_dependents
// CHECK: tosa.const
// CHECK: tosa.transpose
// CHECK: %[[CEIL:.*]] = tosa.ceil
// CHECK: %[[ADD:.*]] = tosa.add %[[CEIL]]
// CHECK: %[[RES1:.*]] = tosa.transpose %[[CEIL]]
// CHECK: %[[RES2:.*]] = tosa.transpose %[[ADD]]
// CHECK: return %[[RES1]], %[[RES2]]
func.func @test_two_same_perms_fan_in_but_one_doesnt_convert_dependents(%arg0: tensor<2x3xi32>, %arg1: tensor<3x2xi32>) -> (tensor<2x3xi32>, tensor<2x3xi32>) {
  %0 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
  %1 = tosa.transpose %arg0, %0 : (tensor<2x3xi32>, tensor<2xi32>) -> tensor<3x2xi32>
  %2 = tosa.ceil %1 : (tensor<3x2xi32>) -> tensor<3x2xi32>
  %3 = tosa.add %2, %arg1 : (tensor<3x2xi32>, tensor<3x2xi32>) -> tensor<3x2xi32>
  %4 = tosa.transpose %2, %0 : (tensor<3x2xi32>, tensor<2xi32>) -> tensor<2x3xi32>
  %5 = tosa.transpose %3, %0 : (tensor<3x2xi32>, tensor<2xi32>) -> tensor<2x3xi32>
  return %4, %5 : tensor<2x3xi32>, tensor<2x3xi32>
}

// -----

// CHECK-LABEL: @test_direct_use_in_other_transpose_with_same_perms
// CHECK-NEXT: %[[RES:.*]] = tosa.clamp %arg0
// CHECK-NEXT: return %[[RES]], %[[RES]]
func.func @test_direct_use_in_other_transpose_with_same_perms(%arg0: tensor<3x3x3x3xi32>) -> (tensor<3x3x3x3xi32>, tensor<3x3x3x3xi32>) {
  %perms0 = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %0 = tosa.transpose %arg0, %perms0 : (tensor<3x3x3x3xi32>, tensor<4xi32>) -> tensor<3x3x3x3xi32>
  %clamp = tosa.clamp %0 {max_val = 2147483647 : i32, min_val = 0 : i32} : (tensor<3x3x3x3xi32>) -> tensor<3x3x3x3xi32>
  %perms1 = "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  %1 = tosa.transpose %clamp, %perms1 : (tensor<3x3x3x3xi32>, tensor<4xi32>) -> tensor<3x3x3x3xi32>
  %2 = tosa.transpose %clamp, %perms1 : (tensor<3x3x3x3xi32>, tensor<4xi32>) -> tensor<3x3x3x3xi32>
  return %1, %2 : tensor<3x3x3x3xi32>, tensor<3x3x3x3xi32>
}

// -----

// CHECK-LABEL: @test_const_transpose
// CHECK: %[[NEW:.*]] = "tosa.const"() <{value = dense<0> : tensor<2x3xi32>}> : () -> tensor<2x3xi32>
// CHECK-NOT: tosa.transpose
// CHECK: return %[[NEW]]
func.func @test_const_transpose() -> tensor<2x3xi32> {
  %0 = "tosa.const"() {value = dense<0> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
  %perms = "tosa.const"() {value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
  %1 = tosa.transpose %0, %perms : (tensor<3x2xi32>, tensor<2xi32>) -> tensor<2x3xi32>
  return %1 : tensor<2x3xi32>
}

// -----

// CHECK-LABEL: @test_transpose_tracks_to_const_single_step
// CHECK: %[[NEW_CONST:.*]] = "tosa.const"() <{value = dense<0> : tensor<1x2x3x4xi32>}> : () -> tensor<1x2x3x4xi32>
// CHECK: %[[NEW_CLAMP:.*]] = tosa.clamp %[[NEW_CONST]] {max_val = 2147483647 : i32, min_val = 0 : i32} : (tensor<1x2x3x4xi32>) -> tensor<1x2x3x4xi32>
// CHECK-NOT: tosa.transpose
// CHECK: return %[[NEW_CLAMP]]
func.func @test_transpose_tracks_to_const_single_step() -> tensor<1x2x3x4xi32> {
  %0 = "tosa.const"() {value = dense<0> : tensor<1x3x4x2xi32>} : () -> tensor<1x3x4x2xi32>
  %clamp = tosa.clamp %0 {max_val = 2147483647 : i32, min_val = 0 : i32} : (tensor<1x3x4x2xi32>) -> tensor<1x3x4x2xi32>
  %perms1 = "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  %1 = tosa.transpose %clamp, %perms1 : (tensor<1x3x4x2xi32>, tensor<4xi32>) -> tensor<1x2x3x4xi32>
  return %1 : tensor<1x2x3x4xi32>
}

// -----

// CHECK-LABEL: @test_static_unary_path_to_const
// CHECK: %[[NEW_CONST:.*]] = "tosa.const"() <{value = dense<1> : tensor<1x2x3x4xi32>}> : () -> tensor<1x2x3x4xi32>
// CHECK: %[[NEW_CLAMP:.*]] = tosa.clamp %[[NEW_CONST]] {max_val = 2147483647 : i32, min_val = 0 : i32} : (tensor<1x2x3x4xi32>) -> tensor<1x2x3x4xi32>
// CHECK: %[[NEW_ABS:.*]] = tosa.abs %[[NEW_CLAMP]] : (tensor<1x2x3x4xi32>) -> tensor<1x2x3x4xi32>
// CHECK: %[[NEW_NOT:.*]] = tosa.bitwise_not %[[NEW_ABS]] : (tensor<1x2x3x4xi32>) -> tensor<1x2x3x4xi32>
// CHECK: return %[[NEW_NOT]]
func.func @test_static_unary_path_to_const() -> tensor<1x2x3x4xi32> {
  %perms0 = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %0 = "tosa.const"() {value = dense<1> : tensor<1x3x4x2xi32>} : () -> tensor<1x3x4x2xi32>
  %clamp = tosa.clamp %0 {max_val = 2147483647 : i32, min_val = 0 : i32} : (tensor<1x3x4x2xi32>) -> tensor<1x3x4x2xi32>
  %abs = tosa.abs %clamp : (tensor<1x3x4x2xi32>) -> tensor<1x3x4x2xi32>
  %bitwise_not = tosa.bitwise_not %abs : (tensor<1x3x4x2xi32>) -> tensor<1x3x4x2xi32>
  %perms1 = "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  %1 = tosa.transpose %bitwise_not, %perms1 : (tensor<1x3x4x2xi32>, tensor<4xi32>) -> tensor<1x2x3x4xi32>
  return %1 : tensor<1x2x3x4xi32>
}

// -----

// CHECK-LABEL: @test_static_diverges_to_non_splat_const_and_nullifying
// CHECK: %[[NEW_CONST:.*]] = "tosa.const"()
// CHECK-SAME{LITERAL}: dense<[[[[1, 3, 5, 7], [9, 11, 13, 15], [17, 19, 21, 23]], [[2, 4, 6, 8], [10, 12, 14, 16], [18, 20, 22, 24]]]]>
// CHECK: tensor<1x2x3x4xi32>}> : () -> tensor<1x2x3x4xi32>
// CHECK: %[[NEW_CLAMP:.*]] = tosa.clamp %arg0 {max_val = 2147483647 : i32, min_val = 0 : i32} : (tensor<1x2x3x4xi32>) -> tensor<1x2x3x4xi32>
// CHECK: %[[NEW_ABS:.*]] = tosa.abs %[[NEW_CONST]] : (tensor<1x2x3x4xi32>) -> tensor<1x2x3x4xi32>
// CHECK: %[[NEW_ADD:.*]] = tosa.add %[[NEW_ABS]], %[[NEW_CLAMP]] : (tensor<1x2x3x4xi32>, tensor<1x2x3x4xi32>) -> tensor<1x2x3x4xi32>
// CHECK: return %[[NEW_ADD]]
func.func @test_static_diverges_to_non_splat_const_and_nullifying(%arg0: tensor<1x2x3x4xi32>) -> tensor<1x2x3x4xi32> {
  %perms0 = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %transpose0 = tosa.transpose %arg0, %perms0 : (tensor<1x2x3x4xi32>, tensor<4xi32>) -> tensor<1x3x4x2xi32>
  %const = "tosa.const"() {value = dense<[[[[1, 2], [3, 4], [5, 6], [7, 8]],
   [[9, 10], [11, 12], [13, 14], [15, 16]],
   [[17, 18], [19, 20], [21, 22], [23, 24]]]]> : tensor<1x3x4x2xi32>} : () -> tensor<1x3x4x2xi32>
  %clamp = tosa.clamp %transpose0 {max_val = 2147483647 : i32, min_val = 0 : i32} : (tensor<1x3x4x2xi32>) -> tensor<1x3x4x2xi32>
  %abs = tosa.abs %const : (tensor<1x3x4x2xi32>) -> tensor<1x3x4x2xi32>
  %add = tosa.add %abs, %clamp : (tensor<1x3x4x2xi32>, tensor<1x3x4x2xi32>) -> tensor<1x3x4x2xi32>
  %perms2 = "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  %result = tosa.transpose %add, %perms2 : (tensor<1x3x4x2xi32>, tensor<4xi32>) -> tensor<1x2x3x4xi32>
  return %result : tensor<1x2x3x4xi32>
}

// -----

// CHECK-LABEL: @test_multi_downstream_both_nullify
// CHECK-NEXT: %[[RES:.*]] = tosa.clamp %arg0
// CHECK-NEXT: return %[[RES]], %[[RES]]
func.func @test_multi_downstream_both_nullify(%arg0: tensor<3x3x3x3xi32>) -> (tensor<3x3x3x3xi32>, tensor<3x3x3x3xi32>) {
  %perms0 = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %0 = tosa.transpose %arg0, %perms0 : (tensor<3x3x3x3xi32>, tensor<4xi32>) -> tensor<3x3x3x3xi32>
  %clamp = tosa.clamp %0 {max_val = 2147483647 : i32, min_val = 0 : i32} : (tensor<3x3x3x3xi32>) -> tensor<3x3x3x3xi32>
  %perms1 = "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  %1 = tosa.transpose %clamp, %perms1 : (tensor<3x3x3x3xi32>, tensor<4xi32>) -> tensor<3x3x3x3xi32>
  %2 = tosa.transpose %clamp, %perms1 : (tensor<3x3x3x3xi32>, tensor<4xi32>) -> tensor<3x3x3x3xi32>
  return %1, %2 : tensor<3x3x3x3xi32>, tensor<3x3x3x3xi32>
}

// -----

// COM: we don't perform this transformation intentionally, since we would then get duplicates
// CHECK-LABEL: @test_multi_downstream_one_nullifies_upstream_other_does_not
// CHECK: tosa.const
// CHECK: tosa.transpose
// CHECK: tosa.clamp
// CHECK: tosa.const
// CHECK: tosa.transpose
// CHECK: tosa.transpose
func.func @test_multi_downstream_one_nullifies_upstream_other_does_not(%arg0: tensor<3x3x3x3xi32>) -> (tensor<3x3x3x3xi32>, tensor<3x3x3x3xi32>) {
  %perms0 = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %0 = tosa.transpose %arg0, %perms0 : (tensor<3x3x3x3xi32>, tensor<4xi32>) -> tensor<3x3x3x3xi32>
  %clamp = tosa.clamp %0 {max_val = 2147483647 : i32, min_val = 0 : i32} : (tensor<3x3x3x3xi32>) -> tensor<3x3x3x3xi32>
  %perms1 = "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  %1 = tosa.transpose %clamp, %perms1 : (tensor<3x3x3x3xi32>, tensor<4xi32>) -> tensor<3x3x3x3xi32>
  %2 = tosa.transpose %clamp, %perms0 : (tensor<3x3x3x3xi32>, tensor<4xi32>) -> tensor<3x3x3x3xi32>
  return %1, %2 : tensor<3x3x3x3xi32>, tensor<3x3x3x3xi32>
}

// -----

// CHECK-LABEL: @test_unknown_dim_inner_replacement_matches
// CHECK-NEXT: return %arg0
func.func @test_unknown_dim_inner_replacement_matches(%arg0: tensor<3x2xi32>) -> tensor<3x2xi32> {
  %perms = "tosa.const"() {value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
  %0 = tosa.transpose %arg0, %perms : (tensor<3x2xi32>, tensor<2xi32>) -> tensor<?x3xi32>
  %1 = tosa.transpose %0, %perms : (tensor<?x3xi32>, tensor<2xi32>) -> tensor<3x2xi32>
  return %1 : tensor<3x2xi32>
}

// -----


// CHECK-LABEL: @test_unknown_dim_outer_replacement_matches
// CHECK-NEXT: return %arg0
func.func @test_unknown_dim_outer_replacement_matches(%arg0: tensor<3x?xi32>) -> tensor<3x?xi32> {
  %perms = "tosa.const"() {value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
  %0 = tosa.transpose %arg0, %perms : (tensor<3x?xi32>, tensor<2xi32>) -> tensor<2x3xi32>
  %1 = tosa.transpose %0, %perms : (tensor<2x3xi32>, tensor<2xi32>) -> tensor<3x?xi32>
  return %1 : tensor<3x?xi32>
}

// -----

// CHECK-LABEL: @test_transpose_tracks_to_nullifying_diverging_binary_unknown_dim_replacements_match
// CHECK-NEXT: %[[CLAMP:.*]] = tosa.clamp %arg0
// CHECK-NEXT: %[[ABS:.*]] = tosa.abs %arg1
// CHECK-NEXT: %[[ADD:.*]] = tosa.add %[[CLAMP]], %[[ABS]]
// CHECK-NEXT: return %[[ADD]]
func.func @test_transpose_tracks_to_nullifying_diverging_binary_unknown_dim_replacements_match(%arg0: tensor<1x?x3x4xi32>, %arg1: tensor<1x2x?x4xi32>) -> tensor<1x2x3x4xi32> {
  %perms0 = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %transpose0 = tosa.transpose %arg0, %perms0 : (tensor<1x?x3x4xi32>, tensor<4xi32>) -> tensor<?x3x4x?xi32>
  %transpose1 = tosa.transpose %arg1, %perms0 : (tensor<1x2x?x4xi32>, tensor<4xi32>) -> tensor<1x?x?x2xi32>
  %clamp = tosa.clamp %transpose0 {min_val = 0 : i32, max_val = 1 : i32} : (tensor<?x3x4x?xi32>) -> tensor<?x3x4x?xi32>
  %abs = tosa.abs %transpose1 : (tensor<1x?x?x2xi32>) -> tensor<1x?x?x2xi32>
  %add = tosa.add %clamp, %abs : (tensor<?x3x4x?xi32>, tensor<1x?x?x2xi32>) -> tensor<1x3x4x2xi32>
  %perms1 = "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  %result = tosa.transpose %add, %perms1 : (tensor<1x3x4x2xi32>, tensor<4xi32>) -> tensor<1x2x3x4xi32>
  return %result : tensor<1x2x3x4xi32>
}

// -----

// COM: we cannot do anything to the transpose in this case.
// CHECK-LABEL: @test_unimplemented_non_const_perms
// CHECK: tosa.const
// CHECK-NEXT: tosa.transpose
// CHECK-NEXT: return
func.func @test_unimplemented_non_const_perms(%perms: tensor<2xi32>) -> tensor<?x?xi32> {
  %0 = "tosa.const"() {value = dense<0> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
  %1 = tosa.transpose %0, %perms : (tensor<3x2xi32>, tensor<2xi32>) -> tensor<?x?xi32>
  return %1 : tensor<?x?xi32>
}

// -----

// COM: due to tracking back to a non-nullifying transpose, we can't get rid of the transposes entirely.
// COM: later editions of the pass may wish to fold these into a single transpose.
// CHECK-LABEL: @test_unimplemented_transpose_tracks_to_non_nullifying_transpose_single_step
// CHECK: tosa.const
// CHECK-NEXT: tosa.transpose
// CHECK-NEXT: tosa.clamp
// CHECK-NEXT: tosa.const
// CHECK-NEXT: tosa.transpose
// CHECK-NEXT: return
func.func @test_unimplemented_transpose_tracks_to_non_nullifying_transpose_single_step(%arg0: tensor<1x2x3x4xi32>) -> tensor<1x2x4x3xi32> {
  %perms0 = "tosa.const"() {value = dense<[0, 3, 2, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %0 = tosa.transpose %arg0, %perms0 : (tensor<1x2x3x4xi32>, tensor<4xi32>) -> tensor<1x4x3x2xi32>
  %clamp = tosa.clamp %0 {min_val = 0 : i32, max_val = 1 : i32} : (tensor<1x4x3x2xi32>) -> tensor<1x4x3x2xi32>
  %perms1 = "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  %1 = tosa.transpose %clamp, %perms1 : (tensor<1x4x3x2xi32>, tensor<4xi32>) -> tensor<1x2x4x3xi32>
  return %1 : tensor<1x2x4x3xi32>
}

// -----

// COM: we don't deal with this case. resolution of shapes required.
// CHECK-LABEL: @test_unimplemented_unknown_dim_input_nullifying_pair
// CHECK: tosa.const
// CHECK-NEXT: tosa.transpose
// CHECK-NEXT: tosa.transpose
// CHECK-NEXT: return
func.func @test_unimplemented_unknown_dim_input_nullifying_pair(%arg0: tensor<3x?xi32>) -> tensor<3x2xi32> {
  %perms = "tosa.const"() {value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
  %0 = tosa.transpose %arg0, %perms : (tensor<3x?xi32>, tensor<2xi32>) -> tensor<2x3xi32>
  %1 = tosa.transpose %0, %perms : (tensor<2x3xi32>, tensor<2xi32>) -> tensor<3x2xi32>
  return %1 : tensor<3x2xi32>
}

// -----

// CHECK-LABEL: @test_unimplemented_unknown_dim_replacement_does_not_match
// CHECK: tosa.const
// CHECK-NEXT: tosa.transpose
// CHECK-NEXT: tosa.transpose
// CHECK-NEXT: return
func.func @test_unimplemented_unknown_dim_replacement_does_not_match(%arg0: tensor<3x?xi32>) -> tensor<?x?xi32> {
  %perms = "tosa.const"() {value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
  %0 = tosa.transpose %arg0, %perms : (tensor<3x?xi32>, tensor<2xi32>) -> tensor<?x3xi32>
  %1 = tosa.transpose %0, %perms : (tensor<?x3xi32>, tensor<2xi32>) -> tensor<?x?xi32>
  return %1 : tensor<?x?xi32>
}

// -----

// COM: this would be able to be converted if --tosa-infer-shapes was run beforehand
// CHECK-LABEL: @test_unimplemented_unranked_tensors_present
// CHECK: tosa.const
// CHECK-NEXT: tosa.transpose
// CHECK-NEXT: tosa.transpose
// CHECK-NEXT: return
func.func @test_unimplemented_unranked_tensors_present(%arg0: tensor<3x2xi32>) -> tensor<*xi32> {
  %perms = "tosa.const"() {value = dense<[0, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
  %0 = tosa.transpose %arg0, %perms : (tensor<3x2xi32>, tensor<2xi32>) -> tensor<*xi32>
  %1 = tosa.transpose %0, %perms : (tensor<*xi32>, tensor<2xi32>) -> tensor<*xi32>
  return %1 : tensor<*xi32>
}

// -----

// CHECK-LABEL: @test_unimplemented_unranked_everything
// CHECK: tosa.const
// CHECK-NEXT: tosa.transpose
// CHECK-NEXT: tosa.transpose
// CHECK-NEXT: return
func.func @test_unimplemented_unranked_everything(%arg0: tensor<*xi32>) -> tensor<*xi32> {
  %perms = "tosa.const"() {value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
  %0 = tosa.transpose %arg0, %perms : (tensor<*xi32>, tensor<2xi32>) -> tensor<*xi32>
  %1 = tosa.transpose %0, %perms : (tensor<*xi32>, tensor<2xi32>) -> tensor<*xi32>
  return %1 : tensor<*xi32>
}

// -----

// CHECK-LABEL: @test_unimplemented_static_diverges_to_one_nullifying_one_non_nullifying
// CHECK: tosa.const
// CHECK-NEXT: tosa.const
// CHECK-NEXT: tosa.transpose
// CHECK-NEXT: tosa.transpose
// CHECK-NEXT: tosa.clamp
// CHECK-NEXT: tosa.abs
// CHECK-NEXT: tosa.add
// CHECK-NEXT: tosa.const
// CHECK-NEXT: tosa.transpose
// CHECK-NEXT: return
func.func @test_unimplemented_static_diverges_to_one_nullifying_one_non_nullifying(%arg0: tensor<1x2x3x4xi32>, %arg1: tensor<1x2x4x3xi32>) -> tensor<1x2x3x4xi32> {
  %perms0 = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %perms1 = "tosa.const"() {value = dense<[0, 3, 2, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %transpose0 = tosa.transpose %arg0, %perms0 : (tensor<1x2x3x4xi32>, tensor<4xi32>) -> tensor<1x3x4x2xi32>
  %transpose1 = tosa.transpose %arg1, %perms1 : (tensor<1x2x4x3xi32>, tensor<4xi32>) -> tensor<1x3x4x2xi32>
  %clamp = tosa.clamp %transpose0 {min_val = 0 : i32, max_val = 1 : i32} : (tensor<1x3x4x2xi32>) -> tensor<1x3x4x2xi32>
  %abs = tosa.abs %transpose1 : (tensor<1x3x4x2xi32>) -> tensor<1x3x4x2xi32>
  %add = tosa.add %clamp, %abs : (tensor<1x3x4x2xi32>, tensor<1x3x4x2xi32>) -> tensor<1x3x4x2xi32>
  %perms2 = "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  %result = tosa.transpose %add, %perms2 : (tensor<1x3x4x2xi32>, tensor<4xi32>) -> tensor<1x2x3x4xi32>
  return %result : tensor<1x2x3x4xi32>
}
