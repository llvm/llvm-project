// RUN: mlir-opt --split-input-file -pass-pipeline="builtin.module(func.func(tosa-to-linalg))" %s -verify-diagnostics -o -| FileCheck %s

// CHECK: #[[$MAP0:.*]] = affine_map<() -> ()>

// CHECK-LABEL: @test_abs_scalar
// CHECK-SAME: ([[ARG0:%[0-9a-zA-Z_]*]]
func.func @test_abs_scalar(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: [[INIT:%.+]] = tensor.empty() : tensor<f32>
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP0]]], iterator_types = []} ins([[ARG0]] : tensor<f32>) outs([[INIT]] : tensor<f32>) {
  // CHECK:   ^bb0([[ARG1:%.*]]: f32, [[ARG2:%.*]]: f32):
  // CHECK:   [[ELEMENT:%.*]] = math.absf [[ARG1]] : f32
  // CHECK:   linalg.yield [[ELEMENT]] : f32
  // CHECK: } -> tensor<f32>
  %0 = tosa.abs %arg0 : (tensor<f32>) -> tensor<f32>

  // CHECK: return [[GENERIC]] : tensor<f32>
	return %0 : tensor<f32>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: @test_abs_1d_cast_static_to_dynamic
// CHECK-SAME: ([[ARG0:%[0-9a-zA-Z_]*]]
func.func @test_abs_1d_cast_static_to_dynamic(%arg0: tensor<5xf32>) -> tensor<?xf32> {
  // CHECK: [[EMPTY:%.+]] = tensor.empty() : tensor<5xf32>
  // CHECK: [[RESULT:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP0]]], iterator_types = ["parallel"]} ins([[ARG0]] : tensor<5xf32>) outs([[EMPTY]] : tensor<5xf32>) {
  // CHECK: ^bb0([[IN0:%.+]]: f32, [[OUT0:%.+]]: f32):
  // CHECK:   [[ABS:%.+]] = math.absf [[IN0]] : f32
  // CHECK:   linalg.yield [[ABS]] : f32
  // CHECK: } -> tensor<5xf32>
  // CHECK: [[CAST_RESULT:%.+]] = tensor.cast [[RESULT]] : tensor<5xf32> to tensor<?xf32>
  %0 = "tosa.abs"(%arg0) : (tensor<5xf32>) -> tensor<?xf32>

  // CHECK: return [[CAST_RESULT]] : tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: @test_abs_1d_cast_dynamic_to_static
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]
func.func @test_abs_1d_cast_dynamic_to_static(%arg0: tensor<?xf32>) -> tensor<5xf32> {
  // CHECK: %[[ZERO:.*]] = arith.constant 0 : index
  // CHECK: %[[DIM_SIZE:.*]] = tensor.dim %[[ARG0]], %[[ZERO]] : tensor<?xf32>
  // CHECK: %[[EMPTY:.*]] = tensor.empty(%[[DIM_SIZE]]) : tensor<?xf32>
  // CHECK: %[[RESULT:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP0]]], iterator_types = ["parallel"]} ins(%[[ARG0]] : tensor<?xf32>) outs(%[[EMPTY]] : tensor<?xf32>) {
  // CHECK: ^bb0(%[[VAL_0:.*]]: f32, %[[VAL_1:.*]]: f32):
  // CHECK:   %[[VAL_2:.*]] = math.absf %[[VAL_0]] : f32
  // CHECK:   linalg.yield %[[VAL_2]] : f32
  // CHECK: } -> tensor<?xf32>
  // CHECK: %[[CAST_RESULT:.*]] = tensor.cast %[[RESULT]] : tensor<?xf32> to tensor<5xf32>
  %0 = "tosa.abs"(%arg0) : (tensor<?xf32>) -> tensor<5xf32>

  // CHECK: return %[[CAST_RESULT]] : tensor<5xf32>
  return %0 : tensor<5xf32>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: @test_abs_1d_dynamic
// CHECK-SAME: ([[ARG0:%[0-9a-zA-Z_]*]]
func.func @test_abs_1d_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {

  // CHECK: [[ZERO:%.+]] = arith.constant 0 : index
  // CHECK: [[DIM:%.+]] = tensor.dim [[ARG0]], [[ZERO]] : tensor<?xf32>
  // CHECK: [[EMPTY:%.+]] = tensor.empty([[DIM]]) : tensor<?xf32>
  // CHECK: [[RESULT:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg0 : tensor<?xf32>) outs([[EMPTY]] : tensor<?xf32>) {
  // CHECK: ^bb0([[IN0:%.+]]: f32, [[OUT0:%.+]]: f32):
  // CHECK:   [[ABSF:%.+]] = math.absf [[IN0]] : f32
  // CHECK:   linalg.yield [[ABSF]] : f32
  // CHECK: } -> tensor<?xf32>
  %0 = tosa.abs %arg0 : (tensor<?xf32>) -> tensor<?xf32>

  // CHECK: return [[RESULT]] : tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<() -> ()>
// CHECK-LABEL: @test_add_0d
// CHECK-SAME: [[ARG0:%[0-9a-zA-Z_]*]]:
// CHECK-SAME: [[ARG1:%[0-9a-zA-Z_]*]]:
func.func @test_add_0d(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {

  // CHECK: [[EMPTY:%.+]] = tensor.empty() : tensor<f32>
  // CHECK: [[RESULT:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins([[ARG0]], [[ARG1]] : tensor<f32>, tensor<f32>) outs([[EMPTY]] : tensor<f32>) {
  // CHECK: ^bb0([[IN0:%.+]]: f32, [[IN1:%.+]]: f32, [[OUT0:%.+]]: f32):
  // CHECK:   [[ADDF:%.+]] = arith.addf [[IN0]], [[IN1]] : f32
  // CHECK:   linalg.yield [[ADDF]] : f32
  // CHECK: } -> tensor<f32>
  %0 = tosa.add %arg0, %arg1 : (tensor<f32>, tensor<f32>) -> tensor<f32>


  // CHECK: return [[RESULT]] : tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK: #[[$MAP0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1) -> (0, d1)>

// CHECK-LABEL:   func.func @test_add_2d_broadcast(
// CHECK-SAME:                                     %[[ARG0:.*]]: tensor<2x1xf32>,
// CHECK-SAME:                                     %[[ARG1:.*]]: tensor<1x1xf32>) -> tensor<2x1xf32> {
// CHECK:           %[[EMPTY_TENSOR:.*]] = tensor.empty() : tensor<2x1xf32>
// CHECK:           %[[RESULT:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP0]]], iterator_types = ["parallel", "parallel"]} ins(%[[ARG0]], %[[ARG1]] : tensor<2x1xf32>, tensor<1x1xf32>) outs(%[[EMPTY_TENSOR]] : tensor<2x1xf32>) {
// CHECK:           ^bb0(%[[IN0:.*]]: f32, %[[IN1:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK:             %[[ADD:.*]] = arith.addf %[[IN0]], %[[IN1]] : f32
// CHECK:             linalg.yield %[[ADD]] : f32
// CHECK:           } -> tensor<2x1xf32>
// CHECK:           return %[[RESULT]] : tensor<2x1xf32>
// CHECK:         }
func.func @test_add_2d_broadcast(%arg0: tensor<2x1xf32>, %arg1: tensor<1x1xf32>) -> tensor<2x1xf32> {
  // tosa element-wise operators now require operands of equal ranks
  %0 = tosa.add %arg0, %arg1 : (tensor<2x1xf32>, tensor<1x1xf32>) -> tensor<2x1xf32>
  return %0 : tensor<2x1xf32>
}

// -----

// CHECK: #[[$MAP0:.+]] = affine_map<(d0) -> (0)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: @test_add_1d_all_dynamic
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z_]*]]:
// CHECK-SAME: %[[ARG1:[0-9a-zA-Z_]*]]:
func.func @test_add_1d_all_dynamic(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {

  // CHECK: %[[CONST0:.*]] = arith.constant 0 : index
  // CHECK: %[[ARG0_DIM0:.*]] = tensor.dim %[[ARG0]], %[[CONST0]] : tensor<?xf32>
  // CHECK: %[[ARG1_DIM0:.*]] = tensor.dim %[[ARG1]], %[[CONST0]] : tensor<?xf32>
  // CHECK: %[[ARG0_MAX_DIM:.*]] = arith.maxui %[[ARG0_DIM0]], %[[ARG1_DIM0]] : index
  // CHECK: %[[CONST1:.*]] = arith.constant 1 : index
  // CHECK: %[[VAL_0:.*]] = tensor.dim %[[ARG0]], %[[CONST0]] : tensor<?xf32>
  // CHECK: %[[VAL_1:.*]] = arith.cmpi eq, %[[VAL_0]], %[[CONST1]] : index
  // CHECK: %[[ARG0_DIM0_BROADCAST:.*]] = scf.if %[[VAL_1]] -> (tensor<?xf32>) {
  // CHECK:   %[[VAL_2:.*]] = tensor.empty(%[[ARG0_MAX_DIM]]) : tensor<?xf32>
  // CHECK:   %[[VAL_3:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel"]} ins(%[[ARG0]] : tensor<?xf32>) outs(%[[VAL_2]] : tensor<?xf32>) {
  // CHECK:   ^bb0(%[[VAL_4:.*]]: f32, %[[VAL_5:.*]]: f32):
  // CHECK:     linalg.yield %[[VAL_4]] : f32
  // CHECK:   } -> tensor<?xf32>
  // CHECK:   scf.yield %[[VAL_3]] : tensor<?xf32>
  // CHECK: } else {
  // CHECK:   scf.yield %[[ARG0]] : tensor<?xf32>
  // CHECK: }
  // CHECK: %[[VAL_6:.*]] = tensor.dim %[[ARG1]], %[[CONST0]] : tensor<?xf32>
  // CHECK: %[[VAL_7:.*]] = arith.cmpi eq, %[[VAL_6]], %[[CONST1]] : index
  // CHECK: %[[ARG0_DIM1_BROADCAST:.*]] = scf.if %[[VAL_7]] -> (tensor<?xf32>) {
  // CHECK:   %[[VAL_8:.*]] = tensor.empty(%[[ARG0_MAX_DIM]]) : tensor<?xf32>
  // CHECK:   %[[VAL_9:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel"]} ins(%[[ARG1]] : tensor<?xf32>) outs(%[[VAL_8]] : tensor<?xf32>) {
  // CHECK:   ^bb0(%[[VAL_10:.*]]: f32, %[[VAL_11:.*]]: f32):
  // CHECK:     linalg.yield %[[VAL_10]] : f32
  // CHECK:   } -> tensor<?xf32>
  // CHECK:   scf.yield %[[VAL_9]] : tensor<?xf32>
  // CHECK: } else {
  // CHECK:   scf.yield %[[ARG1]] : tensor<?xf32>
  // CHECK: }
  // CHECK: %[[VAL_12:.*]] = tensor.empty(%[[ARG0_MAX_DIM]]) : tensor<?xf32>
  // CHECK: %[[RESULT:.*]] = linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel"]} ins(%[[ARG0_DIM0_BROADCAST]], %[[ARG0_DIM1_BROADCAST]] : tensor<?xf32>, tensor<?xf32>) outs(%[[VAL_12]] : tensor<?xf32>) {
  // CHECK: ^bb0(%[[VAL_13:.*]]: f32, %[[VAL_14:.*]]: f32, %[[VAL_15:.*]]: f32):
  // CHECK:   %[[VAL_16:.*]] = arith.addf %[[VAL_13]], %[[VAL_14]] : f32
  // CHECK:   linalg.yield %[[VAL_16]] : f32
  // CHECK: } -> tensor<?xf32>
  %0 = tosa.add %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>

  // CHECK: return %[[RESULT]] : tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

// CHECK: #[[$MAP0:.+]] = affine_map<(d0) -> (0)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: @test_add_1d_broadcast_dynamic_to_static
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z_]*]]:
// CHECK-SAME: %[[ARG1:[0-9a-zA-Z_]*]]:
func.func @test_add_1d_broadcast_dynamic_to_static(%arg0: tensor<5xf32>, %arg1: tensor<?xf32>) -> tensor<5xf32> {

  // CHECK: %[[CONST1:.*]] = arith.constant 1 : index
  // CHECK: %[[CONST0:.*]] = arith.constant 0 : index
  // CHECK: %[[ARG1_DIM0:.*]] = tensor.dim %[[ARG1]], %[[CONST0]] : tensor<?xf32>
  // CHECK: %[[VAL_0:.*]] = arith.cmpi eq, %[[ARG1_DIM0]], %[[CONST1]] : index
  // CHECK: %[[ARG1_DIM0_BROADCAST:.*]] = scf.if %[[VAL_0]] -> (tensor<?xf32>) {
  // CHECK:   %[[VAL_1:.*]] = tensor.empty() : tensor<5xf32>
  // CHECK:   %[[VAL_2:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel"]} ins(%[[ARG1]] : tensor<?xf32>) outs(%[[VAL_1]] : tensor<5xf32>) {
  // CHECK:   ^bb0(%[[VAL_3:.*]]: f32, %[[VAL_4:.*]]: f32):
  // CHECK:     linalg.yield %[[VAL_3]] : f32
  // CHECK:   } -> tensor<5xf32>
  // CHECK:   %[[VAL_5:.*]] = tensor.cast %[[VAL_2]] : tensor<5xf32> to tensor<?xf32>
  // CHECK:   scf.yield %[[VAL_5]] : tensor<?xf32>
  // CHECK: } else {
  // CHECK:   scf.yield %[[ARG1]] : tensor<?xf32>
  // CHECK: }
  // CHECK: %[[VAL_6:.*]] = tensor.empty() : tensor<5xf32>
  // CHECK: %[[RESULT:.*]] = linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel"]} ins(%[[ARG0]], %[[ARG1_DIM0_BROADCAST]] : tensor<5xf32>, tensor<?xf32>) outs(%[[VAL_6]] : tensor<5xf32>) {
  // CHECK: ^bb0(%[[VAL_7:.*]]: f32, %[[VAL_8:.*]]: f32, %[[VAL_9:.*]]: f32):
  // CHECK:   %[[VAL_10:.*]] = arith.addf %[[VAL_7]], %[[VAL_8]] : f32
  // CHECK:   linalg.yield %[[VAL_10]] : f32
  // CHECK: } -> tensor<5xf32>
  %0 = tosa.add %arg0, %arg1 : (tensor<5xf32>, tensor<?xf32>) -> tensor<5xf32>

  // CHECK: return %[[RESULT]] : tensor<5xf32>
  return %0 : tensor<5xf32>
}

// -----

// CHECK: #[[$MAP0:.+]] = affine_map<(d0) -> (0)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: @test_add_1d_broadcast_static_to_dynamic
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z_]*]]:
// CHECK-SAME: %[[ARG1:[0-9a-zA-Z_]*]]:
func.func @test_add_1d_broadcast_static_to_dynamic(%arg0: tensor<1xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {

  // CHECK: %[[CONST0:.*]] = arith.constant 0 : index
  // CHECK: %[[ARG1_DIM0:.*]] = tensor.dim %[[ARG1]], %[[CONST0]] : tensor<?xf32>
  // CHECK: %[[VAL_0:.*]] = tensor.empty(%[[ARG1_DIM0]]) : tensor<?xf32>
  // CHECK: %[[RESULT:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel"]} ins(%[[ARG0]], %[[ARG1]] : tensor<1xf32>, tensor<?xf32>) outs(%[[VAL_0]] : tensor<?xf32>) {
  // CHECK: ^bb0(%[[VAL_1:.*]]: f32, %[[VAL_2:.*]]: f32, %[[VAL_3:.*]]: f32):
  // CHECK:   %[[VAL_4:.*]] = arith.addf %[[VAL_1]], %[[VAL_2]] : f32
  // CHECK:   linalg.yield %[[VAL_4]] : f32
  // CHECK: } -> tensor<?xf32>
  %0 = tosa.add %arg0, %arg1 : (tensor<1xf32>, tensor<?xf32>) -> tensor<?xf32>

  // CHECK: return %[[RESULT]] : tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

// CHECK: #[[$MAP0:.+]] = affine_map<(d0) -> (0)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: @test_add_1d_broadcast_static_to_static
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z_]*]]:
// CHECK-SAME: %[[ARG1:[0-9a-zA-Z_]*]]:
func.func @test_add_1d_broadcast_static_to_static(%arg0: tensor<1xf32>, %arg1: tensor<3xf32>) -> tensor<3xf32> {

  // CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<3xf32>
  // CHECK: %[[RESULT:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel"]} ins(%[[ARG0]], %[[ARG1]] : tensor<1xf32>, tensor<3xf32>) outs(%[[VAL_0]] : tensor<3xf32>) {
  // CHECK: ^bb0(%[[VAL_1:.*]]: f32, %[[VAL_2:.*]]: f32, %[[VAL_3:.*]]: f32):
  // CHECK:   %[[VAL_4:.*]] = arith.addf %[[VAL_1]], %[[VAL_2]] : f32
  // CHECK:   linalg.yield %[[VAL_4]] : f32
  // CHECK: } -> tensor<3xf32>
  %0 = tosa.add %arg0, %arg1 : (tensor<1xf32>, tensor<3xf32>) -> tensor<3xf32>

  // CHECK: return %[[RESULT]] : tensor<3xf32>
  return %0 : tensor<3xf32>
}

// -----

// CHECK: #[[$MAP:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: @test_add_1d_matching_no_broadcast
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z_]*]]:
// CHECK-SAME: %[[ARG1:[0-9a-zA-Z_]*]]:
func.func @test_add_1d_matching_no_broadcast(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {

  // CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<1xf32>
  // CHECK: %[[RESULT:.*]] = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP]], #[[$MAP]]], iterator_types = ["parallel"]} ins(%[[ARG0]], %[[ARG1]] : tensor<1xf32>, tensor<1xf32>) outs(%[[VAL_0]] : tensor<1xf32>) {
  // CHECK: ^bb0(%[[VAL_1:.*]]: f32, %[[VAL_2:.*]]: f32, %[[VAL_3:.*]]: f32):
  // CHECK:   %[[VAL_4:.*]] = arith.addf %[[VAL_1]], %[[VAL_2]] : f32
  // CHECK:   linalg.yield %[[VAL_4]] : f32
  // CHECK: } -> tensor<1xf32>
  %0 = tosa.add %arg0, %arg1 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>

  // CHECK: return %[[RESULT]] : tensor<1xf32>
  return %0 : tensor<1xf32>
}

// -----

// CHECK: #[[$MAP0:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: @test_add_1d_matching_static
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z_]*]]:
// CHECK-SAME: %[[ARG1:[0-9a-zA-Z_]*]]:
func.func @test_add_1d_matching_static(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> tensor<3xf32> {

  // CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<3xf32>
  // CHECK: %[[RESULT:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP0]], #[[$MAP0]]], iterator_types = ["parallel"]} ins(%[[ARG0]], %[[ARG1]] : tensor<3xf32>, tensor<3xf32>) outs(%[[VAL_0]] : tensor<3xf32>) {
  // CHECK: ^bb0(%[[VAL_1:.*]]: f32, %[[VAL_2:.*]]: f32, %[[VAL_3:.*]]: f32):
  // CHECK:   %[[VAL_4:.*]] = arith.addf %[[VAL_1]], %[[VAL_2]] : f32
  // CHECK:   linalg.yield %[[VAL_4]] : f32
  // CHECK: } -> tensor<3xf32>
  %0 = tosa.add %arg0, %arg1 : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>

  // CHECK: return %[[RESULT]] : tensor<3xf32>
  return %0 : tensor<3xf32>
}

// -----

// CHECK: #[[$MAP0:.+]] = affine_map<(d0, d1) -> (0, d1)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1) -> (d0, 0)>
// CHECK-LABEL: @test_add_2d_all_dynamic
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z_]*]]:
// CHECK-SAME: %[[ARG1:[0-9a-zA-Z_]*]]:
func.func @test_add_2d_all_dynamic(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {

  // CHECK: %[[CONST0:.*]] = arith.constant 0 : index
  // CHECK: %[[ARG0_DIM0:.*]] = tensor.dim %[[ARG0]], %[[CONST0]] : tensor<?x?xf32>
  // CHECK: %[[ARG1_DIM0:.*]] = tensor.dim %[[ARG1]], %[[CONST0]] : tensor<?x?xf32>
  // CHECK: %[[MAX_DIM0:.*]] = arith.maxui %[[ARG0_DIM0]], %[[ARG1_DIM0]] : index
  // CHECK: %[[CONST1:.*]] = arith.constant 1 : index
  // CHECK: %[[ARG0_DIM1:.*]] = tensor.dim %[[ARG0]], %[[CONST1]] : tensor<?x?xf32>
  // CHECK: %[[ARG1_DIM1:.*]] = tensor.dim %[[ARG1]], %[[CONST1]] : tensor<?x?xf32>
  // CHECK: %[[MAX_DIM1:.*]] = arith.maxui %[[ARG0_DIM1]], %[[ARG1_DIM1]] : index

  // CHECK: %[[VAL_0:.*]] = tensor.dim %[[ARG0]], %[[CONST0]] : tensor<?x?xf32>
  // CHECK: %[[VAL_1:.*]] = arith.cmpi eq, %[[VAL_0]], %[[CONST1]] : index
  // CHECK: %[[ARG0_DIM0_BROADCAST:.*]] = scf.if %[[VAL_1]] -> (tensor<?x?xf32>) {
  // CHECK:   %[[LOCAL_CONST1:.*]] = arith.constant 1 : index
  // CHECK:   %[[VAL_2:.*]] = tensor.dim %[[ARG0]], %[[LOCAL_CONST1]] : tensor<?x?xf32>
  // CHECK:   %[[VAL_3:.*]] = tensor.empty(%[[MAX_DIM0]], %[[VAL_2]]) : tensor<?x?xf32>
  // CHECK:   %[[VAL_4:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel"]} ins(%[[ARG0]] : tensor<?x?xf32>) outs(%[[VAL_3]] : tensor<?x?xf32>) {
  // CHECK:   ^bb0(%[[VAL_5:.*]]: f32, %[[VAL_6:.*]]: f32):
  // CHECK:     linalg.yield %[[VAL_5]] : f32
  // CHECK:   } -> tensor<?x?xf32>
  // CHECK:   scf.yield %[[VAL_4]] : tensor<?x?xf32>
  // CHECK: } else {
  // CHECK:   scf.yield %[[ARG0]] : tensor<?x?xf32>
  // CHECK: }

  // CHECK: %[[VAL_7:.*]] = tensor.dim %[[ARG0_DIM0_BROADCAST]], %[[CONST1]] : tensor<?x?xf32>
  // CHECK: %[[VAL_8:.*]] = arith.cmpi eq, %[[VAL_7]], %[[CONST1]] : index
  // CHECK: %[[ARG0_DIM1_BROADCAST:.*]] = scf.if %[[VAL_8]] -> (tensor<?x?xf32>) {
  // CHECK:   %[[LOCAL_CONST0:.*]] = arith.constant 0 : index
  // CHECK:   %[[VAL_9:.*]] = tensor.dim %[[ARG0_DIM0_BROADCAST]], %[[LOCAL_CONST0]] : tensor<?x?xf32>
  // CHECK:   %[[VAL_10:.*]] = tensor.empty(%[[VAL_9]], %[[MAX_DIM1]]) : tensor<?x?xf32>
  // CHECK:   %[[VAL_11:.*]] = linalg.generic {indexing_maps = [#[[$MAP2]], #[[$MAP1]]], iterator_types = ["parallel", "parallel"]} ins(%[[ARG0_DIM0_BROADCAST]] : tensor<?x?xf32>) outs(%[[VAL_10]] : tensor<?x?xf32>) {
  // CHECK:   ^bb0(%[[VAL_12:.*]]: f32, %[[VAL_13:.*]]: f32):
  // CHECK:     linalg.yield %[[VAL_12]] : f32
  // CHECK:   } -> tensor<?x?xf32>
  // CHECK:   scf.yield %[[VAL_11]] : tensor<?x?xf32>
  // CHECK: } else {
  // CHECK:   scf.yield %[[ARG0_DIM0_BROADCAST]] : tensor<?x?xf32>
  // CHECK: }

  // CHECK: %[[VAL_14:.*]] = tensor.dim %[[ARG1]], %[[CONST0]] : tensor<?x?xf32>
  // CHECK: %[[VAL_15:.*]] = arith.cmpi eq, %[[VAL_14]], %[[CONST1]] : index
  // CHECK: %[[ARG1_DIM0_BROADCAST:.*]] = scf.if %[[VAL_15]] -> (tensor<?x?xf32>) {
  // CHECK:   %[[LOCAL_CONST1:.*]] = arith.constant 1 : index
  // CHECK:   %[[VAL_16:.*]] = tensor.dim %[[ARG1]], %[[LOCAL_CONST1]] : tensor<?x?xf32>
  // CHECK:   %[[VAL_17:.*]] = tensor.empty(%[[MAX_DIM0]], %[[VAL_16]]) : tensor<?x?xf32>
  // CHECK:   %[[VAL_18:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel"]} ins(%[[ARG1]] : tensor<?x?xf32>) outs(%[[VAL_17]] : tensor<?x?xf32>) {
  // CHECK:   ^bb0(%[[VAL_19:.*]]: f32, %[[VAL_20:.*]]: f32):
  // CHECK:     linalg.yield %[[VAL_19]] : f32
  // CHECK:   } -> tensor<?x?xf32>
  // CHECK:   scf.yield %[[VAL_18]] : tensor<?x?xf32>
  // CHECK: } else {
  // CHECK:   scf.yield %[[ARG1]] : tensor<?x?xf32>
  // CHECK: }

  // CHECK: %[[VAL_21:.*]] = tensor.dim %[[ARG1_DIM0_BROADCAST]], %[[CONST1]] : tensor<?x?xf32>
  // CHECK: %[[VAL_22:.*]] = arith.cmpi eq, %[[VAL_21]], %[[CONST1]] : index
  // CHECK: %[[ARG1_DIM1_BROADCAST:.*]] = scf.if %[[VAL_22]] -> (tensor<?x?xf32>) {
  // CHECK:   %[[LOCAL_CONST0:.*]] = arith.constant 0 : index
  // CHECK:   %[[VAL_23:.*]] = tensor.dim %[[ARG1_DIM0_BROADCAST]], %[[LOCAL_CONST0]] : tensor<?x?xf32>
  // CHECK:   %[[VAL_24:.*]] = tensor.empty(%[[VAL_23]], %[[MAX_DIM1]]) : tensor<?x?xf32>
  // CHECK:   %[[VAL_25:.*]] = linalg.generic {indexing_maps = [#[[$MAP2]], #[[$MAP1]]], iterator_types = ["parallel", "parallel"]} ins(%[[ARG1_DIM0_BROADCAST]] : tensor<?x?xf32>) outs(%[[VAL_24]] : tensor<?x?xf32>) {
  // CHECK:   ^bb0(%[[VAL_26:.*]]: f32, %[[VAL_27:.*]]: f32):
  // CHECK:     linalg.yield %[[VAL_26]] : f32
  // CHECK:   } -> tensor<?x?xf32>
  // CHECK:   scf.yield %[[VAL_25]] : tensor<?x?xf32>
  // CHECK: } else {
  // CHECK:   scf.yield %[[ARG1_DIM0_BROADCAST]] : tensor<?x?xf32>
  // CHECK: }

  // CHECK: %[[VAL_28:.*]] = tensor.empty(%[[MAX_DIM0]], %[[MAX_DIM1]]) : tensor<?x?xf32>
  // CHECK: %[[RESULT:.*]] = linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel", "parallel"]} ins(%[[ARG0_DIM1_BROADCAST]], %[[ARG1_DIM1_BROADCAST]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[VAL_28]] : tensor<?x?xf32>) {
  // CHECK: ^bb0(%[[VAL_29:.*]]: f32, %[[VAL_30:.*]]: f32, %[[VAL_31:.*]]: f32):
  // CHECK:   %[[VAL_32:.*]] = arith.addf %[[VAL_29]], %[[VAL_30]] : f32
  // CHECK:   linalg.yield %[[VAL_32]] : f32
  // CHECK: } -> tensor<?x?xf32>
  %0 = tosa.add %arg0, %arg1 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>

  // CHECK: return %[[RESULT]] : tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// CHECK: #[[$MAP0:.+]] = affine_map<(d0, d1) -> (d0, 0)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: @test_select_2d_one_dynamic
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z_]*]]:
// CHECK-SAME: %[[ARG1:[0-9a-zA-Z_]*]]:
// CHECK-SAME: %[[ARG2:[0-9a-zA-Z_]*]]:
func.func @test_select_2d_one_dynamic(%arg0: tensor<2x?xi1>, %arg1: tensor<2x?xf32>, %arg2: tensor<2x?xf32>) -> tensor<2x?xf32> {

  // CHECK: %[[CONST1:.*]] = arith.constant 1 : index
  // CHECK: %[[ARG0_DIM1:.*]] = tensor.dim %[[ARG0]], %[[CONST1]] : tensor<2x?xi1>
  // CHECK: %[[ARG1_DIM1:.*]] = tensor.dim %[[ARG1]], %[[CONST1]] : tensor<2x?xf32>
  // CHECK: %[[VAL_0:.*]] = arith.maxui %[[ARG0_DIM1]], %[[ARG1_DIM1]] : index
  // CHECK: %[[ARG2_DIM1:.*]] = tensor.dim %[[ARG2]], %[[CONST1]] : tensor<2x?xf32>
  // CHECK: %[[MAX_DIM1:.*]] = arith.maxui %[[VAL_0]], %[[ARG2_DIM1]] : index

  // CHECK: %[[VAL_1:.*]] = tensor.dim %[[ARG0]], %[[CONST1]] : tensor<2x?xi1>
  // CHECK: %[[VAL_2:.*]] = arith.cmpi eq, %[[VAL_1]], %[[CONST1]] : index
  // CHECK: %[[ARG0_BROADCAST:.*]] = scf.if %[[VAL_2]] -> (tensor<2x?xi1>) {
  // CHECK:   %[[VAL_3:.*]] = tensor.empty(%[[MAX_DIM1]]) : tensor<2x?xi1>
  // CHECK:   %[[VAL_4:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel"]} ins(%[[ARG0]] : tensor<2x?xi1>) outs(%[[VAL_3]] : tensor<2x?xi1>) {
  // CHECK:   ^bb0(%[[VAL_5:.*]]: i1, %[[VAL_6:.*]]: i1):
  // CHECK:     linalg.yield %[[VAL_5]] : i1
  // CHECK:   } -> tensor<2x?xi1>
  // CHECK:   scf.yield %[[VAL_4]] : tensor<2x?xi1>
  // CHECK: } else {
  // CHECK:   scf.yield %[[ARG0]] : tensor<2x?xi1>
  // CHECK: }

  // CHECK: %[[VAL_7:.*]] = tensor.dim %[[ARG1]], %[[CONST1]] : tensor<2x?xf32>
  // CHECK: %[[VAL_8:.*]] = arith.cmpi eq, %[[VAL_7]], %[[CONST1]] : index
  // CHECK: %[[ARG1_BROADCAST:.*]] = scf.if %[[VAL_8]] -> (tensor<2x?xf32>) {
  // CHECK:   %[[VAL_9:.*]] = tensor.empty(%[[MAX_DIM1]]) : tensor<2x?xf32>
  // CHECK:   %[[VAL_10:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel"]} ins(%[[ARG1]] : tensor<2x?xf32>) outs(%[[VAL_9]] : tensor<2x?xf32>) {
  // CHECK:   ^bb0(%[[VAL_11:.*]]: f32, %[[VAL_12:.*]]: f32):
  // CHECK:     linalg.yield %[[VAL_11]] : f32
  // CHECK:   } -> tensor<2x?xf32>
  // CHECK:   scf.yield %[[VAL_10]] : tensor<2x?xf32>
  // CHECK: } else {
  // CHECK:   scf.yield %[[ARG1]] : tensor<2x?xf32>
  // CHECK: }

  // CHECK: %[[VAL_13:.*]] = tensor.dim %[[ARG2]], %[[CONST1]] : tensor<2x?xf32>
  // CHECK: %[[VAL_14:.*]] = arith.cmpi eq, %[[VAL_13]], %[[CONST1]] : index
  // CHECK: %[[ARG2_BROADCAST:.*]] = scf.if %[[VAL_14]] -> (tensor<2x?xf32>) {
  // CHECK:   %[[VAL_15:.*]] = tensor.empty(%[[MAX_DIM1]]) : tensor<2x?xf32>
  // CHECK:   %[[VAL_16:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel"]} ins(%[[ARG2]] : tensor<2x?xf32>) outs(%[[VAL_15]] : tensor<2x?xf32>) {
  // CHECK:   ^bb0(%[[VAL_17:.*]]: f32, %[[VAL_18:.*]]: f32):
  // CHECK:     linalg.yield %[[VAL_17]] : f32
  // CHECK:   } -> tensor<2x?xf32>
  // CHECK:   scf.yield %[[VAL_16]] : tensor<2x?xf32>
  // CHECK: } else {
  // CHECK:   scf.yield %[[ARG2]] : tensor<2x?xf32>
  // CHECK: }

  // CHECK: %[[VAL_19:.*]] = tensor.empty(%[[MAX_DIM1]]) : tensor<2x?xf32>
  // CHECK: %[[RESULT:.*]] = linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP1]], #[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel", "parallel"]} ins(%[[ARG0_BROADCAST]], %[[ARG1_BROADCAST]], %[[ARG2_BROADCAST]] : tensor<2x?xi1>, tensor<2x?xf32>, tensor<2x?xf32>) outs(%[[VAL_19]] : tensor<2x?xf32>) {
  // CHECK: ^bb0(%[[VAL_20:.*]]: i1, %[[VAL_21:.*]]: f32, %[[VAL_22:.*]]: f32, %[[VAL_23:.*]]: f32):
  // CHECK:   %[[VAL_24:.*]] = arith.select %[[VAL_20]], %[[VAL_21]], %[[VAL_22]] : f32
  // CHECK:   linalg.yield %[[VAL_24]] : f32
  // CHECK: } -> tensor<2x?xf32>
  %0 = tosa.select %arg0, %arg1, %arg2 : (tensor<2x?xi1>, tensor<2x?xf32>, tensor<2x?xf32>) -> tensor<2x?xf32>

  // CHECK: return %[[RESULT]] : tensor<2x?xf32>
  return %0 : tensor<2x?xf32>
}

// -----

// CHECK-LABEL: @test_simple_f32
func.func @test_simple_f32(%arg0: tensor<1xf32>) -> () {
  // CHECK: linalg.generic
  // CHECK: tanh
  %0 = tosa.tanh %arg0 : (tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: math.absf
  %1 = tosa.abs %arg0 : (tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: arith.addf
  %2 = tosa.add %0, %0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: arith.subf
  %3 = tosa.sub %0, %1 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: arith.mulf
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %4 = tosa.mul %0, %1, %shift : (tensor<1xf32>, tensor<1xf32>, tensor<1xi8>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: arith.negf
  %in_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %out_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %5 = tosa.negate %0, %in_zp, %out_zp : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: pow
  %6 = tosa.pow %1, %2 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: rsqrt
  %7 = tosa.rsqrt %1 : (tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: log
  %8 = tosa.log %arg0 : (tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: exp
  %9 = tosa.exp %arg0 : (tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: arith.cmpf
  %10 = tosa.greater %0, %1 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>

  // CHECK: linalg.generic
  // CHECK: arith.cmpf
  %11 = tosa.greater_equal %0, %1 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>

  // CHECK: linalg.generic
  // CHECK: arith.cmpf
  %12 = tosa.equal %0, %1 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>

  // CHECK: linalg.generic
  // CHECK: select
  %13 = tosa.select %10, %0, %1 : (tensor<1xi1>, tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: arith.maximumf
  %14 = tosa.maximum %0, %1 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: arith.minimumf
  %15 = tosa.minimum %0, %1 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: ceil
  %16 = tosa.ceil %0 : (tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: floor
  %17 = tosa.floor %0 : (tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: arith.minimumf
  // CHECK: arith.maximumf
  %18 = tosa.clamp %0 {min_val = 1.0 : f32, max_val = 5.0 : f32} : (tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: arith.negf
  // CHECK: exp
  // CHECK: arith.addf
  // CHECK: arith.divf
  %19 = tosa.sigmoid %0 : (tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: [[ROUND:%.+]] = math.roundeven {{%.+}} : f32
  // CHECK: [[CSTMIN:%.+]] = arith.constant -2.14748365E+9 : f32
  // CHECK: [[CSTMAXP1:%.+]] = arith.constant 2.14748365E+9 : f32
  // CHECK: [[CSTMAX:%.+]] = arith.constant 2147483647 : i32
  // CHECK: [[MAX:%.+]] = arith.maximumf [[ROUND]], [[CSTMIN]] : f32
  // CHECK: [[CONV:%.+]] = arith.fptosi [[MAX]] : f32 to i32
  // CHECK: [[CMP:%.+]] = arith.cmpf uge, [[ROUND]], [[CSTMAXP1]] : f32
  // CHECK: arith.select [[CMP]], [[CSTMAX]], [[CONV]] : i32
  %20 = tosa.cast %0 : (tensor<1xf32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: arith.constant 0
  // CHECK: arith.cmpf
  %21 = tosa.cast %0 : (tensor<1xf32>) -> tensor<1xi1>

  // CHECK: linalg.generic
  // CHECK: arith.truncf
  %22 = tosa.cast %0 : (tensor<1xf32>) -> tensor<1xf16>

  // CHECK: linalg.generic
  // CHECK: arith.divf
  %23 = tosa.reciprocal %0 : (tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: math.erf
  %24 = tosa.erf %0 : (tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: math.sin
  %25 = tosa.sin %arg0 : (tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: math.cos
  %26 = tosa.cos %arg0 : (tensor<1xf32>) -> tensor<1xf32>

  return
}

// -----

// CHECK-LABEL: @test_simple_f16
func.func @test_simple_f16(%arg0: tensor<1xf16>) -> () {

  // CHECK: linalg.generic
  // CHECK: arith.extf
  %0 = tosa.cast %arg0 : (tensor<1xf16>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: [[ROUND:%.+]] = math.roundeven {{%.+}} : f16
  // CHECK: [[CSTMIN:%.+]] = arith.constant -1.280000e+02 : f16
  // CHECK: [[CSTMAX:%.+]] = arith.constant 1.270000e+02 : f16
  // CHECK: [[MIN:%.+]] = arith.minimumf [[ROUND]], [[CSTMAX]] : f16
  // CHECK: [[CLAMP:%.+]] = arith.maximumf [[MIN]], [[CSTMIN]] : f16
  // CHECK: arith.fptosi [[CLAMP]] : f16 to i8
  %1 = "tosa.cast"(%arg0) : (tensor<1xf16>) -> tensor<1xi8>

  // CHECK: linalg.generic
  // CHECK: [[ROUND:%.+]] = math.roundeven {{%[a-z0-9_]+}} : f16
  // CHECK: [[CONV:%.+]] = arith.fptosi [[ROUND]] : f16 to i32
  // CHECK: [[POSINF:%.+]] = arith.constant 0x7C00 : f16
  // CHECK: [[NEGINF:%.+]] = arith.constant 0xFC00 : f16
  // CHECK: [[OVERFLOW:%.+]] = arith.cmpf ueq, [[ROUND]], [[POSINF]] : f16
  // CHECK: [[UNDERFLOW:%.+]] = arith.cmpf ueq, [[ROUND]], [[NEGINF]] : f16
  // CHECK: [[MININT:%.+]] = arith.constant -2147483648 : i32
  // CHECK: [[MAXINT:%.+]] = arith.constant 2147483647 : i32
  // CHECK: [[CLAMPPOSINF:%.+]] = arith.select [[OVERFLOW]], [[MAXINT]], [[CONV]] : i32
  // CHECK: arith.select [[UNDERFLOW]], [[MININT]], [[CLAMPPOSINF]] : i32
  %2 = "tosa.cast"(%arg0) : (tensor<1xf16>) -> tensor<1xi32>
  return
}

// -----

// CHECK-LABEL: @test_simple_i16
func.func @test_simple_i16(%arg0: tensor<1xi16>) -> () {
  // CHECK: linalg.generic
  // CHECK: arith.extsi
  // CHECK: arith.extsi
  // CHECK: arith.muli
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %0 = tosa.mul %arg0, %arg0, %shift : (tensor<1xi16>, tensor<1xi16>, tensor<1xi8>) -> tensor<1xi32>

  return
}

// -----

// CHECK-LABEL: @test_simple_ui8
func.func @test_simple_ui8(%arg0: tensor<1xui8>) -> () {
  // CHECK: arith.uitofp
  %0 = tosa.cast %arg0 : (tensor<1xui8>) -> tensor<1xf32>
  return
}

// -----

// CHECK-LABEL: @test_simple_i32
func.func @test_simple_i32(%arg0: tensor<1xi32>, %unsigned: tensor<1xui32>, %unsigned64: tensor<1xui64>) -> () {
  // CHECK: linalg.generic
  // CHECK: arith.addi
  %0 = tosa.add %arg0, %arg0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: arith.subi
  %1 = tosa.sub %arg0, %arg0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: arith.muli
  %shift1 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %2 = tosa.mul %arg0, %arg0, %shift1 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi8>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: arith.constant 2
  // CHECK: apply_scale
  %shift2 = "tosa.const"() <{values = dense<2> : tensor<1xi8>}> : () -> tensor<1xi8>
  %3 = tosa.mul %arg0, %arg0, %shift2: (tensor<1xi32>, tensor<1xi32>, tensor<1xi8>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: arith.divsi
  %4 = tosa.intdiv %arg0, %arg0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: ^bb0(%[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32):
  // CHECK: [[ZERO:%.+]] = arith.constant 0
  // CHECK: arith.subi [[ZERO]], %[[ARG1]]
  %in_zp = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
  %out_zp = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
  %5 = tosa.negate %arg0, %in_zp, %out_zp : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: and
  %6 = tosa.bitwise_and %arg0, %arg0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: or
  %7 = tosa.bitwise_or %arg0, %arg0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: arith.xori
  %8 = tosa.bitwise_xor %arg0, %arg0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: arith.shli
  %9 = tosa.logical_left_shift %arg0, %arg0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: arith.shrui
  %10 = tosa.logical_right_shift %arg0, %arg0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: arith.shrsi
  %11 = tosa.arithmetic_right_shift %arg0, %arg0 {round = 0 : i1} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: arith.constant 1
  // CHECK: arith.constant 0
  // CHECK: arith.constant true
  // CHECK: arith.cmpi
  // CHECK: arith.subi
  // CHECK: arith.shrsi
  // CHECK: arith.trunci
  // CHECK: and
  // CHECK: and
  // CHECK: arith.extui
  // CHECK: arith.addi
  %12 = tosa.arithmetic_right_shift %arg0, %arg0 {round = 1 : i1} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  // CHECK: math.ctlz
  %13 = tosa.clz %arg0 : (tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: arith.cmpi
  %14 = tosa.greater %0, %1 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>

  // CHECK: linalg.generic
  // CHECK: arith.cmpi
  %15 = tosa.greater_equal %0, %1 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>

  // CHECK: linalg.generic
  // CHECK: select
  %16 = tosa.select %14, %0, %1 : (tensor<1xi1>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: arith.maxsi
  %17 = tosa.maximum %0, %1 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: arith.minsi
  %18 = tosa.minimum %0, %1 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK-DAG: arith.maxsi
  // CHECK-DAG: arith.minsi
  %19 = tosa.clamp %0 {min_val = 1 : i32, max_val = 5 : i32} : (tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK-DAG: %[[LB:.*]] = arith.constant 4 : i32
  // CHECK-DAG: %[[UB:.*]] = arith.constant 32 : i32
  // CHECK-DAG: arith.maxui %[[LB]],
  // CHECK-DAG: arith.minui %[[UB]],
  %u0 = tosa.clamp %unsigned {min_val = 4 : ui32, max_val = 32 : ui32} : (tensor<1xui32>) -> tensor<1xui32>

  // CHECK: linalg.generic
  // CHECK: arith.trunci
  %20 = tosa.cast %0 : (tensor<1xi32>) -> tensor<1xi16>

  // CHECK: linalg.generic
  // CHECK: arith.extsi
  %21 = tosa.cast %0 : (tensor<1xi32>) -> tensor<1xi64>

  // CHECK: linalg.generic
  // CHECK: arith.constant 0
  // CHECK: arith.cmpi
  %22 = tosa.cast %0 : (tensor<1xi32>) -> tensor<1xi1>

  // CHECK: linalg.generic
  // CHECK: arith.sitofp
  %23 = tosa.cast %0 : (tensor<1xi32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: arith.constant 0
  // CHECK: arith.subi
  // CHECK: arith.maxsi
  %24 = tosa.abs %arg0 : (tensor<1xi32>) -> tensor<1xi32>

  return
}

// -----

// CHECK-LABEL: @test_simple_ui8
func.func @test_simple_ui8(%arg0: tensor<1xi8>) -> () {

  // CHECK: linalg.generic
  // CHECK: sitofp
  %0 = tosa.cast %arg0 : (tensor<1xi8>) -> tensor<1xf32>

  return
}

// -----

// CHECK-LABEL: @test_i8
func.func @test_i8(%arg0: tensor<1xi8>) -> () {
  // CHECK: linalg.generic
  // CHECK: ^bb0(%[[ARG1:.+]]: i8,
  // CHECK-DAG: %[[C127:.+]] = arith.constant -127
  // CHECK-DAG: %[[C126:.+]] = arith.constant 126
  // CHECK-DAG: %[[LOWER:.+]] = arith.maxsi %[[C127]], %[[ARG1]]
  // CHECK-DAG: %[[CLAMPED:.+]] = arith.minsi %[[C126]], %[[LOWER]]
  %0 = tosa.clamp %arg0 {min_val = -127 : i8, max_val = 126 : i8} : (tensor<1xi8>) -> tensor<1xi8>

  return
}

// -----

// CHECK-LABEL: @test_i64
func.func @test_i64(%arg0: tensor<1xi64>) -> () {
  // CHECK: linalg.generic
  // CHECK: ^bb0(%[[ARG1:.+]]: i64,
  // CHECK-DAG: %[[C127:.+]] = arith.constant -9223372036854775808
  // CHECK-DAG: %[[C126:.+]] = arith.constant 9223372036854775807
  // CHECK-DAG: %[[LOWER:.+]] = arith.maxsi %[[C127]], %[[ARG1]]
  // CHECK-DAG: %[[CLAMPED:.+]] = arith.minsi %[[C126]], %[[LOWER]]
  %0 = tosa.clamp %arg0 {min_val = -9223372036854775808 : i64, max_val = 9223372036854775807 : i64} : (tensor<1xi64>) -> tensor<1xi64>

  return
}

// -----

// CHECK-LABEL: @test_clamp_f16
func.func @test_clamp_f16(%arg0: tensor<1xf16>) -> () {
  // CHECK: linalg.generic
  // CHECK: ^bb0(%[[ARG1:.+]]: f16,
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0.0
  // CHECK-DAG: %[[C6:.+]] = arith.constant 6.0
  // CHECK-DAG: %[[MIN:.+]] = arith.minimumf %[[ARG1]], %[[C6]]
  // CHECK-DAG: %[[MAX:.+]] = arith.maximumf %[[MIN]], %[[C0]]
  %0 = tosa.clamp %arg0 {min_val = 0.0 : f16, max_val = 6.0 : f16} : (tensor<1xf16>) -> tensor<1xf16>

  return
}

// -----

// CHECK-LABEL: @test_bool
func.func @test_bool(%arg0: tensor<1xi1>, %arg1: tensor<1xi1>) -> () {
  // CHECK: linalg.generic
  // CHECK: and
  %0 = tosa.logical_and %arg0, %arg1 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>

  // CHECK: linalg.generic
  // CHECK: or
  %1 = tosa.logical_or %arg0, %arg1 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>

  // CHECK: linalg.generic
  // CHECK: arith.xori
  %2 = tosa.logical_xor %arg0, %arg1 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>

  // CHECK: linalg.generic
  // CHECK: arith.constant true
  // CHECK: arith.xori
  %3 = tosa.logical_not %arg0 : (tensor<1xi1>) -> tensor<1xi1>

  return
}

// -----

// CHECK-LABEL: @test_negate_quantized
func.func @test_negate_quantized(%arg0: tensor<1xi8>) -> () {
  // CHECK: linalg.generic
  // CHECK: ^bb0(%[[BBARG0:.+]]: i8, %[[BBARG1:.+]]: i8
  // CHECK: [[CNST:%.+]] = arith.constant 7
  // CHECK: [[EXT:%.+]] = arith.extsi %[[BBARG0]] : i8 to i16
  // CHECK: [[SUB:%.+]] = arith.subi [[CNST]], [[EXT]]
  // CHECK: [[MIN:%.+]] = arith.constant -128
  // CHECK: [[MAX:%.+]] = arith.constant 127
  // CHECK: [[LBOUND:%.+]] = arith.maxsi [[MIN]], [[SUB]]
  // CHECK: [[UBOUND:%.+]] = arith.minsi [[MAX]], [[LBOUND]]
  // CHECK: [[TRUNC:%.+]] = arith.trunci [[UBOUND]]
  // CHECK: linalg.yield [[TRUNC]]
  %in_zp0 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %out_zp0 = "tosa.const"() <{values = dense<7> : tensor<1xi8>}> : () -> tensor<1xi8>
  %0 = tosa.negate %arg0, %in_zp0, %out_zp0 : (tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>

  // CHECK: linalg.generic
  // CHECK: ^bb0(%[[BBARG0:.+]]: i8, %[[BBARG1:.+]]: i8
  // CHECK: [[C_128:%.+]] = arith.constant -128
  // CHECK: [[EXT:%.+]] = arith.extsi %[[BBARG0]] : i8 to i16
  // CHECK: [[SUB:%.+]] = arith.subi [[C_128]], [[EXT]]
  // CHECK: [[MIN:%.+]] = arith.constant -128
  // CHECK: [[MAX:%.+]] = arith.constant 127
  // CHECK: [[LBOUND:%.+]] = arith.maxsi [[MIN]], [[SUB]]
  // CHECK: [[UBOUND:%.+]] = arith.minsi [[MAX]], [[LBOUND]]
  // CHECK: [[TRUNC:%.+]] = arith.trunci [[UBOUND]]
  // CHECK: linalg.yield [[TRUNC]]
  %in_zp3 = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
  %out_zp3 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %3 = tosa.negate %arg0, %in_zp3, %out_zp3 : (tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>

  // CHECK: linalg.generic
  // CHECK: ^bb0(%[[BBARG0:.+]]: i8,
  // CHECK: [[ZERO:%.+]] = arith.constant 0
  // CHECK: [[SUB:%.+]] = arith.subi [[ZERO]],
  // CHECK: linalg.yield [[SUB]]
  %in_zp4 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %out_zp4 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %4 = tosa.negate %arg0, %in_zp4, %out_zp4 : (tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>

  return
}

// -----

// CHECK-LABEL: @test_identity
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z_]*]]: tensor<1xf32>,
// CHECK-SAME: %[[ARG1:[0-9a-zA-Z_]*]]: tensor<1xi32>
func.func @test_identity(%arg0: tensor<1xf32>, %arg1: tensor<1xi32>) -> (tensor<1xf32>, tensor<1xi32>) {
  %0 = tosa.identity %arg0 : (tensor<1xf32>) -> tensor<1xf32>
  %1 = tosa.identity %arg1 : (tensor<1xi32>) -> tensor<1xi32>

  // CHECK: return %[[ARG0]], %[[ARG1]]
  return %0, %1 : tensor<1xf32>, tensor<1xi32>
}

// -----

// CHECK-LABEL: @reduce_float
// CHECK-SAME: [[ARG0:%.+]]: tensor<5x4xf32>
func.func @reduce_float(%arg0: tensor<5x4xf32>) -> () {
  // CHECK: [[INIT:%.+]] = tensor.empty() : tensor<4xf32>
  // CHECK: [[CST0:%.+]] = arith.constant 0.0
  // CHECK: [[FILL:%.+]] = linalg.fill ins([[CST0]]{{.*}}outs([[INIT]]
  // CHECK: [[REDUCE:%.+]] = linalg.reduce ins([[ARG0]] : tensor<5x4xf32>) outs([[FILL]] : tensor<4xf32>) dimensions = [0]
  // CHECK:  (%[[ARG1:.*]]: f32, %[[ARG2:.*]]: f32) {
  // CHECK:   [[RES:%.+]] = arith.addf %[[ARG1]], %[[ARG2]] : f32
  // CHECK:   linalg.yield [[RES]] : f32
  // CHECK:  }
  // CHECK: tensor.expand_shape [[REDUCE]] {{\[}}[0, 1]] output_shape [1, 4] : tensor<4xf32> into tensor<1x4xf32>
  %0 = tosa.reduce_sum %arg0 {axis = 0 : i32} : (tensor<5x4xf32>) -> tensor<1x4xf32>

  // CHECK: [[INIT:%.+]] = tensor.empty() : tensor<5xf32>
  // CHECK: [[CST0:%.+]] = arith.constant 0.0
  // CHECK: [[FILL:%.+]] = linalg.fill ins([[CST0]]{{.*}}outs([[INIT]]
  // CHECK: [[REDUCE:%.+]] = linalg.reduce ins([[ARG0]] : tensor<5x4xf32>) outs([[FILL]] : tensor<5xf32>) dimensions = [1]
  // CHECK:  (%[[ARG1:.*]]: f32, %[[ARG2:.*]]: f32) {
  // CHECK:   [[RES:%.+]] = arith.addf %[[ARG1]], %[[ARG2]] : f32
  // CHECK:   linalg.yield [[RES]] : f32
  // CHECK:  }
  // CHECK: tensor.expand_shape [[REDUCE]] {{\[}}[0, 1]] output_shape [5, 1] : tensor<5xf32> into tensor<5x1xf32>
  %1 = tosa.reduce_sum %arg0 {axis = 1 : i32} : (tensor<5x4xf32>) -> tensor<5x1xf32>

  // CHECK: arith.constant 1.0
  // CHECK: linalg.fill
  // CHECK: linalg.reduce
  // CHECK: arith.mulf
  %2 = tosa.reduce_product %arg0 {axis = 0 : i32} : (tensor<5x4xf32>) -> tensor<1x4xf32>

  // CHECK: arith.constant 3.40282347E+38 : f32
  // CHECK: linalg.fill
  // CHECK: linalg.reduce
  // CHECK: arith.minimumf
  %3 = tosa.reduce_min %arg0 {axis = 0 : i32} : (tensor<5x4xf32>) -> tensor<1x4xf32>

  // CHECK: arith.constant -3.40282347E+38 : f32
  // CHECK: linalg.fill
  // CHECK: linalg.reduce
  // CHECK: arith.maximumf
  %4 = tosa.reduce_max %arg0 {axis = 0 : i32} : (tensor<5x4xf32>) -> tensor<1x4xf32>
  return
}

// -----

// CHECK-LABEL: @reduce_float_dyn
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z_]*]]: tensor<?x5x4xf32>
func.func @reduce_float_dyn(%arg0: tensor<?x5x4xf32>) -> () {
  // CHECK: %[[C0:.+]] = arith.constant 0
  // CHECK: %[[DYN:.+]] = tensor.dim %[[ARG0]], %[[C0]]
  // CHECK: %[[INIT:.+]] = tensor.empty(%[[DYN]]) : tensor<?x4xf32>
  // CHECK: %[[CST0:.+]] = arith.constant 0.0
  // CHECK: %[[FILL:.+]] = linalg.fill ins(%[[CST0]]{{.*}}outs(%[[INIT]]
  // CHECK: %[[REDUCE:.+]] = linalg.reduce ins(%[[ARG0]] : tensor<?x5x4xf32>) outs(%[[FILL]] : tensor<?x4xf32>) dimensions = [1]
  // CHECK:  (%[[ARG1:.*]]: f32, %[[ARG2:.*]]: f32) {
  // CHECK:   %[[RES:.+]] = arith.addf %[[ARG1]], %[[ARG2]] : f32
  // CHECK:   linalg.yield %[[RES]] : f32
  // CHECK:  }
  // CHECK: %[[C0_0:.+]] = arith.constant 0 : index
  // CHECK: %[[DIM_1:.+]] = tensor.dim %[[REDUCE]], %[[C0_0]] : tensor<?x4xf32>
  // CHECK: %[[C1:.+]] = arith.constant 1 : index
  // CHECK: tensor.expand_shape %[[REDUCE]] {{\[}}[0], [1, 2]] output_shape [%[[DIM_1]], 1, 4] : tensor<?x4xf32> into tensor<?x1x4xf32>
  %0 = tosa.reduce_sum %arg0 {axis = 1 : i32} : (tensor<?x5x4xf32>) -> tensor<?x1x4xf32>
  return
}

// -----

// CHECK-LABEL: @reduce_float_dyn_rank_1
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z_]*]]: tensor<?xf32>
func.func @reduce_float_dyn_rank_1(%arg0: tensor<?xf32>) -> () {
  // CHECK-DAG: %[[INIT:.+]] = tensor.empty() : tensor<f32>
  // CHECK-DAG: %[[CST0:.+]] = arith.constant 0.0
  // CHECK: %[[FILL:.+]] = linalg.fill ins(%[[CST0]]{{.*}}outs(%[[INIT]]
  // CHECK: %[[REDUCE:.+]] = linalg.reduce ins(%[[ARG0]] : tensor<?xf32>) outs(%[[FILL]] : tensor<f32>) dimensions = [0]
  // CHECK:  (%[[ARG1:.*]]: f32, %[[ARG2:.*]]: f32) {
  // CHECK:   %[[RES:.+]] = arith.addf %[[ARG1]], %[[ARG2]] : f32
  // CHECK:   linalg.yield %[[RES]] : f32
  // CHECK:  }
  // CHECK: tensor.expand_shape %[[REDUCE]] {{\[}}] output_shape [1] : tensor<f32> into tensor<1xf32>
  %0 = tosa.reduce_sum %arg0 {axis = 0 : i32} : (tensor<?xf32>) -> tensor<1xf32>
  return
}

// -----

// CHECK-LABEL: @reduce_float_dyn_nonzero_batch
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]:
func.func @reduce_float_dyn_nonzero_batch(%arg0: tensor<5x?x4xf32>) -> () {
  // CHECK: %[[C1:.+]] = arith.constant 1
  // CHECK: %[[DYN:.+]] = tensor.dim %[[ARG0]], %[[C1]]
  // CHECK: %[[INIT:.+]] = tensor.empty(%[[DYN]]) : tensor<5x?xf32>
  // CHECK: %[[CST1:.+]] = arith.constant 1.0
  // CHECK: %[[FILL:.+]] = linalg.fill ins(%[[CST1]]{{.*}}outs(%[[INIT]]
  // CHECK: %[[REDUCE:.+]] = linalg.reduce ins(%[[ARG0]] : tensor<5x?x4xf32>) outs(%[[FILL]] : tensor<5x?xf32>) dimensions = [2]
  // CHECK:  (%[[ARG1:.*]]: f32, %[[ARG2:.*]]: f32) {
  // CHECK:   %[[RES:.+]] = arith.mulf %[[ARG1]], %[[ARG2]] : f32
  // CHECK:   linalg.yield %[[RES]] : f32
  // CHECK:  }
  // CHECK: %[[C1_0:.+]] = arith.constant 1 : index
  // CHECK: %[[DIM_1:.+]] = tensor.dim %[[REDUCE]], %[[C1_0]] : tensor<5x?xf32>
  // CHECK: %[[C1_2:.+]] = arith.constant 1 : index
  // CHECK: tensor.expand_shape %[[REDUCE]] {{\[}}[0], [1, 2]] output_shape [5, %[[DIM_1]], 1] : tensor<5x?xf32> into tensor<5x?x1xf32>
  %0 = tosa.reduce_product %arg0 {axis = 2 : i32} : (tensor<5x?x4xf32>) -> tensor<5x?x1xf32>
  return
}

// -----

// CHECK-LABEL: @reduce_float_dyn_multiple
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]:
func.func @reduce_float_dyn_multiple(%arg0: tensor<?x?xf32>) -> () {
  // CHECK: %[[C0:.+]] = arith.constant 0
  // CHECK: %[[DYN:.+]] = tensor.dim %[[ARG0]], %[[C0]]
  // CHECK: %[[INIT:.+]] = tensor.empty(%[[DYN]])
  // CHECK: %[[CMIN:.+]] = arith.constant -3.40282347E+38
  // CHECK: %[[FILL:.+]] = linalg.fill ins(%[[CMIN]]{{.*}}outs(%[[INIT]]
  // CHECK: %[[REDUCE:.+]] = linalg.reduce ins(%[[ARG0]] : tensor<?x?xf32>) outs(%[[FILL]] : tensor<?xf32>) dimensions = [1]
  // CHECK:  (%[[ARG1:.*]]: f32, %[[ARG2:.*]]: f32) {
  // CHECK:   %[[MAX:.+]] = arith.maximumf %[[ARG1]], %[[ARG2]] : f32
  // CHECK:   linalg.yield %[[MAX]] : f32
  // CHECK:  }
  // CHECK: %[[C0_0:.+]] = arith.constant 0 : index
  // CHECK: %[[DIM_1:.+]] = tensor.dim %[[REDUCE]], %[[C0_0]] : tensor<?xf32>
  // CHECK: %[[C1_2:.+]] = arith.constant 1 : index
  // CHECK: tensor.expand_shape %[[REDUCE]] {{\[}}[0, 1]] output_shape [%[[DIM_1]], 1] : tensor<?xf32> into tensor<?x1xf32>
  %0 = tosa.reduce_max %arg0 {axis = 1 : i32} : (tensor<?x?xf32>) -> tensor<?x1xf32>
  return
}

// -----

// CHECK-LABEL: @reduce_int
// CHECK-SAME: [[ARG0:%.+]]: tensor<5x4xi32>
func.func @reduce_int(%arg0: tensor<5x4xi32>) -> () {
  // CHECK: [[INIT:%.+]] = tensor.empty()
  // CHECK: [[CST0:%.+]] = arith.constant 0
  // CHECK: [[FILL:%.+]] = linalg.fill ins([[CST0]]{{.*}}outs([[INIT]]
  // CHECK: [[REDUCE:%.+]] = linalg.reduce ins([[ARG0]] : tensor<5x4xi32>) outs([[FILL]] : tensor<4xi32>) dimensions = [0]
  // CHECK:  (%[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32) {
  // CHECK:   [[RES:%.+]] = arith.addi %[[ARG1]], %[[ARG2]] : i32
  // CHECK:   linalg.yield [[RES]] : i32
  // CHECK:  }
  // CHECK: tensor.expand_shape [[REDUCE]] {{\[}}[0, 1]] output_shape [1, 4] : tensor<4xi32> into tensor<1x4xi32>
  %0 = tosa.reduce_sum %arg0 {axis = 0 : i32} : (tensor<5x4xi32>) -> tensor<1x4xi32>

  // CHECK: [[INIT:%.+]] = tensor.empty()
  // CHECK: [[CST0:%.+]] = arith.constant 0
  // CHECK: [[FILL:%.+]] = linalg.fill ins([[CST0]]{{.*}}outs([[INIT]]
  // CHECK: [[REDUCE:%.+]] = linalg.reduce ins([[ARG0]] : tensor<5x4xi32>) outs([[FILL]] : tensor<5xi32>) dimensions = [1]
  // CHECK:  (%[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32) {
  // CHECK:   [[RES:%.+]] = arith.addi %[[ARG1]], %[[ARG2]] : i32
  // CHECK:   linalg.yield [[RES]] : i32
  // CHECK:  }
  // CHECK: tensor.expand_shape [[REDUCE]] {{\[}}[0, 1]] output_shape [5, 1] : tensor<5xi32> into tensor<5x1xi32>
  %1 = tosa.reduce_sum %arg0 {axis = 1 : i32} : (tensor<5x4xi32>) -> tensor<5x1xi32>

  // CHECK: arith.constant 1
  // CHECK: linalg.fill
  // CHECK: linalg.reduce
  // CHECK: arith.muli
  %2 = tosa.reduce_product %arg0 {axis = 0 : i32} : (tensor<5x4xi32>) -> tensor<1x4xi32>

  // CHECK: arith.constant 2147483647 : i32
  // CHECK: linalg.fill
  // CHECK: linalg.reduce
  // CHECK: arith.minsi
  %3 = tosa.reduce_min %arg0 {axis = 0 : i32} : (tensor<5x4xi32>) -> tensor<1x4xi32>

  // CHECK: arith.constant -2147483648 : i32
  // CHECK: linalg.fill
  // CHECK: linalg.reduce
  // CHECK: arith.maxsi
  %4 = tosa.reduce_max %arg0 {axis = 0 : i32} : (tensor<5x4xi32>) -> tensor<1x4xi32>
  return
}

// -----

// CHECK-LABEL: @reduce_bool
// CHECK-SAME: [[ARG0:%.+]]: tensor<5x4xi1>
func.func @reduce_bool(%arg0: tensor<5x4xi1>) -> () {
  // CHECK: [[INIT:%.+]] = tensor.empty()
  // CHECK: [[CST0:%.+]] = arith.constant true
  // CHECK: [[FILL:%.+]] = linalg.fill ins([[CST0]]{{.*}}outs([[INIT]]
  // CHECK: [[REDUCE:%.+]] = linalg.reduce ins([[ARG0]] : tensor<5x4xi1>) outs([[FILL]] : tensor<4xi1>) dimensions = [0]
  // CHECK:  (%[[ARG1:[0-9a-zA-Z_]+]]: i1, %[[ARG2:[0-9a-zA-Z_]+]]: i1) {
  // CHECK:   [[RES:%.+]] = arith.andi %[[ARG1]], %[[ARG2]] : i1
  // CHECK:   linalg.yield [[RES]] : i1
  // CHECK:  }
  // CHECK: tensor.expand_shape [[REDUCE]] {{\[}}[0, 1]] output_shape [1, 4] : tensor<4xi1> into tensor<1x4xi1>
  %0 = tosa.reduce_all %arg0 {axis = 0 : i32} : (tensor<5x4xi1>) -> tensor<1x4xi1>

  // CHECK: arith.constant false
  // CHECK: linalg.fill
  // CHECK: linalg.reduce
  // CHECK: or
  %1 = tosa.reduce_any %arg0 {axis = 0 : i32} : (tensor<5x4xi1>) -> tensor<1x4xi1>

  return
}

// -----
// CHECK: #[[$MAP0:.*]] = affine_map<(d0) -> (d0)>

// CHECK-LABEL: @rescale_i8
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]:
func.func @rescale_i8(%arg0 : tensor<2xi8>) -> () {
  // CHECK: [[C0:%.+]] = arith.constant 19689
  // CHECK: [[C1:%.+]] = arith.constant 15
  // CHECK: [[INIT:%.+]] = tensor.empty()
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP0]]], iterator_types = ["parallel"]} ins(%[[ARG0]] : tensor<2xi8>) outs([[INIT]] : tensor<2xi8>)
  // CHECK: ^bb0([[IN:%.+]]: i8, [[UNUSED:%.+]]: i8):
  // CHECK: [[C17:%.+]] = arith.constant 17
  // CHECK: [[C22:%.+]] = arith.constant 22
  // CHECK-DAG: [[IN32:%.+]] = arith.extsi [[IN]]
  // CHECK-DAG: [[IN_ZEROED:%.+]] = arith.subi [[IN32]], [[C17]]
  // CHECK-DAG: [[SCALED:%.+]] = tosa.apply_scale [[IN_ZEROED]], [[C0]], [[C1]] {rounding_mode = #tosa.rounding_mode<SINGLE_ROUND>}
  // CHECK-DAG: [[SCALED_ZEROED:%.+]] = arith.addi [[SCALED]], [[C22]]
  // CHECK-DAG: [[CMIN:%.+]] = arith.constant -128
  // CHECK-DAG: [[CMAX:%.+]] = arith.constant 127
  // CHECK-DAG: [[LOWER:%.+]] = arith.maxsi [[CMIN]], [[SCALED_ZEROED]]
  // CHECK-DAG: [[BOUNDED:%.+]] = arith.minsi [[CMAX]], [[LOWER]]
  // CHECK-DAG: [[TRUNC:%.+]] = arith.trunci [[BOUNDED]]
  // CHECK-DAG: linalg.yield [[TRUNC]]
  %multiplier = "tosa.const"() {values = dense<19689> : tensor<1xi16>} : () -> tensor<1xi16>
  %shift = "tosa.const"() {values = dense<15> : tensor<1xi8>} : () -> tensor<1xi8>
  %input_zp = "tosa.const"() {values = dense<17> : tensor<1xi8>} : () -> tensor<1xi8>
  %output_zp = "tosa.const"() {values = dense<22> : tensor<1xi8>} : () -> tensor<1xi8>
  %0 = tosa.rescale %arg0, %multiplier, %shift, %input_zp, %output_zp {scale32 = false, rounding_mode = #tosa.rounding_mode<SINGLE_ROUND>, per_channel = false, input_unsigned = false, output_unsigned = false} : (tensor<2xi8>, tensor<1xi16>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<2xi8>

  // CHECK: return
  return
}

// -----
// CHECK: #[[$MAP0:.*]] = affine_map<(d0) -> (d0)>

// CHECK-LABEL: @rescale_i8_unsigned_output_explicit
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]:
func.func @rescale_i8_unsigned_output_explicit(%arg0 : tensor<2xi8>) -> () {
  // CHECK: [[C0:%.+]] = arith.constant 19689
  // CHECK: [[C1:%.+]] = arith.constant 15
  // CHECK: [[INIT:%.+]] = tensor.empty()
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP0]]], iterator_types = ["parallel"]} ins(%[[ARG0]] : tensor<2xi8>) outs([[INIT]] : tensor<2xui8>)
  // CHECK: ^bb0([[IN:%.+]]: i8, [[UNUSED:%.+]]: ui8):
  // CHECK-DAG: [[C17:%.+]] = arith.constant 17
  // CHECK-DAG: [[C234:%.+]] = arith.constant 234
  // CHECK-DAG: [[IN32:%.+]] = arith.extsi [[IN]]
  // CHECK-DAG: [[IN_ZEROED:%.+]] = arith.subi [[IN32]], [[C17]]
  // CHECK-DAG: [[SCALED:%.+]] = tosa.apply_scale [[IN_ZEROED]], [[C0]], [[C1]] {rounding_mode = #tosa.rounding_mode<SINGLE_ROUND>}
  // CHECK-DAG: [[SCALED_ZEROED:%.+]] = arith.addi [[SCALED]], [[C234]]
  // CHECK-DAG: [[CMIN:%.+]] = arith.constant 0
  // CHECK-DAG: [[CMAX:%.+]] = arith.constant 255
  // CHECK-DAG: [[LOWER:%.+]] = arith.maxsi [[CMIN]], [[SCALED_ZEROED]]
  // CHECK: [[BOUNDED:%.+]] = arith.minsi [[CMAX]], [[LOWER]]
  // CHECK: [[TRUNC:%.+]] = arith.trunci [[BOUNDED]]
  // CHECK: [[TRUNC_ITOU:%.+]] = builtin.unrealized_conversion_cast [[TRUNC]] : i8 to ui8
  // CHECK: linalg.yield [[TRUNC_ITOU]]
  %multiplier = "tosa.const"() {values = dense<19689> : tensor<1xi16> } : () -> tensor<1xi16>
  %shift = "tosa.const"() {values = dense<15> : tensor<1xi8> } : () -> tensor<1xi8>
  %input_zp = "tosa.const"() {values = dense<17> : tensor<1xi8>} : () -> tensor<1xi8>
  %output_zp = "tosa.const"() {values = dense<-22> : tensor<1xi8>} : () -> tensor<1xi8>
  %1 = tosa.rescale %arg0, %multiplier, %shift, %input_zp, %output_zp {scale32 = false, rounding_mode = #tosa.rounding_mode<SINGLE_ROUND>, per_channel = false, input_unsigned = false, output_unsigned = true} : (tensor<2xi8>, tensor<1xi16>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<2xui8>

  // CHECK: return
  return
}

// -----
// CHECK: #[[$MAP0:.*]] = affine_map<(d0) -> (d0)>

// CHECK-LABEL: @rescale_i8_unsigned_output_implicit
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]:
func.func @rescale_i8_unsigned_output_implicit(%arg0 : tensor<2xi8>) -> () {
  // CHECK: [[C0:%.+]] = arith.constant 19689
  // CHECK: [[C1:%.+]] = arith.constant 15
  // CHECK: [[INIT:%.+]] = tensor.empty()
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP0]]], iterator_types = ["parallel"]} ins(%[[ARG0]] : tensor<2xi8>) outs([[INIT]] : tensor<2xi8>)
  // CHECK: ^bb0([[IN:%.+]]: i8, [[UNUSED:%.+]]: i8):
  // CHECK-DAG: [[C17:%.+]] = arith.constant 17
  // CHECK-DAG: [[C234:%.+]] = arith.constant 234
  // CHECK-DAG: [[IN32:%.+]] = arith.extsi [[IN]]
  // CHECK-DAG: [[IN_ZEROED:%.+]] = arith.subi [[IN32]], [[C17]]
  // CHECK-DAG: [[SCALED:%.+]] = tosa.apply_scale [[IN_ZEROED]], [[C0]], [[C1]] {rounding_mode = #tosa.rounding_mode<SINGLE_ROUND>}
  // CHECK-DAG: [[SCALED_ZEROED:%.+]] = arith.addi [[SCALED]], [[C234]]
  // CHECK-DAG: [[CMIN:%.+]] = arith.constant 0
  // CHECK-DAG: [[CMAX:%.+]] = arith.constant 255
  // CHECK-DAG: [[LOWER:%.+]] = arith.maxsi [[CMIN]], [[SCALED_ZEROED]]
  // CHECK: [[BOUNDED:%.+]] = arith.minsi [[CMAX]], [[LOWER]]
  // CHECK: [[TRUNC:%.+]] = arith.trunci [[BOUNDED]]
  // CHECK-NOT: builtin.unrealized_conversion_cast [[TRUNC]]
  // CHECK: linalg.yield [[TRUNC]]
  %multiplier = "tosa.const"() {values = dense<19689> : tensor<1xi16> } : () -> tensor<1xi16>
  %shift = "tosa.const"() {values = dense<15> : tensor<1xi8> } : () -> tensor<1xi8>
  %input_zp = "tosa.const"() {values = dense<17> : tensor<1xi8>} : () -> tensor<1xi8>
  %output_zp = "tosa.const"() {values = dense<-22> : tensor<1xi8>} : () -> tensor<1xi8>
  %1 = tosa.rescale %arg0, %multiplier, %shift, %input_zp, %output_zp {scale32 = false, rounding_mode = #tosa.rounding_mode<SINGLE_ROUND>, per_channel = false, input_unsigned = false, output_unsigned = true} : (tensor<2xi8>, tensor<1xi16>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<2xi8>

  // CHECK: return
  return
}

// -----
// CHECK: #[[$MAP0:.*]] = affine_map<(d0) -> (d0)>

// CHECK-LABEL: @rescale_i48_unsigned_output_implicit
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]:
func.func @rescale_i48_unsigned_output_implicit(%arg0 : tensor<2xi48>) -> () {
  // CHECK: [[C19689:%.+]] = arith.constant 19689
  // CHECK: [[C15:%.+]] = arith.constant 15
  // CHECK: [[INIT:%.+]] = tensor.empty()
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP0]]], iterator_types = ["parallel"]} ins(%[[ARG0]] : tensor<2xi48>) outs([[INIT]] : tensor<2xi8>)
  // CHECK: ^bb0([[IN:%.+]]: i48, [[UNUSED:%.+]]: i8):
  // CHECK-NOT: builtin.unrealized_conversion_cast [[IN]]
  // CHECK-DAG: [[C0:%.+]] = arith.constant 0
  // CHECK-DAG: [[C234:%.+]] = arith.constant 234
  // CHECK-DAG: [[IN_ZEROED:%.+]] = arith.subi [[IN]], [[C0]]
  // CHECK-DAG: [[SCALED:%.+]] = tosa.apply_scale [[IN_ZEROED]], [[C19689]], [[C15]] {rounding_mode = #tosa.rounding_mode<SINGLE_ROUND>}
  // CHECK-DAG: [[SCALED_ZEROED:%.+]] = arith.addi [[SCALED]], [[C234]]
  // CHECK-DAG: [[CMIN:%.+]] = arith.constant 0
  // CHECK-DAG: [[CMAX:%.+]] = arith.constant 255
  // CHECK-DAG: [[LOWER:%.+]] = arith.maxsi [[CMIN]], [[SCALED_ZEROED]]
  // CHECK-DAG: [[BOUNDED:%.+]] = arith.minsi [[CMAX]], [[LOWER]]
  // CHECK-DAG: [[TRUNC:%.+]] = arith.trunci [[BOUNDED]]
  // CHECK: linalg.yield [[TRUNC]]
  %multiplier = "tosa.const"() {values = dense<19689> : tensor<1xi16> } : () -> tensor<1xi16>
  %shift = "tosa.const"() {values = dense<15> : tensor<1xi8> } : () -> tensor<1xi8>
  %input_zp = "tosa.const"() {values = dense<0> : tensor<1xi48>} : () -> tensor<1xi48>
  %output_zp = "tosa.const"() {values = dense<-22> : tensor<1xi8>} : () -> tensor<1xi8>
  %1 = tosa.rescale %arg0, %multiplier, %shift, %input_zp, %output_zp {scale32 = false, rounding_mode = #tosa.rounding_mode<SINGLE_ROUND>, per_channel = false, input_unsigned = false, output_unsigned = true} : (tensor<2xi48>, tensor<1xi16>, tensor<1xi8>, tensor<1xi48>, tensor<1xi8>) -> tensor<2xi8>

  // CHECK: return
  return
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @rescale_i8_dyn_batch
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]:
func.func @rescale_i8_dyn_batch(%arg0 : tensor<?x2xi8>) -> () {
  %multiplier = "tosa.const"() {values = dense<19689> : tensor<1xi16>} : () -> tensor<1xi16>
  %shift = "tosa.const"() {values = dense<15> : tensor<1xi8>} : () -> tensor<1xi8>
  %input_zp = "tosa.const"() {values = dense<17> : tensor<1xi8>} : () -> tensor<1xi8>
  %output_zp = "tosa.const"() {values = dense<22> : tensor<1xi8>} : () -> tensor<1xi8>
  // CHECK: %[[C0:.+]] = arith.constant 0
  // CHECK: %[[BATCH:.+]] = tensor.dim %[[ARG0]], %[[C0]]
  // CHECK: %[[INIT:.+]] = tensor.empty(%[[BATCH]]) : tensor<?x2xi8>
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP0]]], iterator_types = ["parallel", "parallel"]} ins(%[[ARG0]] : tensor<?x2xi8>) outs(%[[INIT]] : tensor<?x2xi8>)
  %0 = tosa.rescale %arg0, %multiplier, %shift, %input_zp, %output_zp {scale32 = false, rounding_mode = #tosa.rounding_mode<SINGLE_ROUND>, per_channel = false, input_unsigned = false, output_unsigned = false} : (tensor<?x2xi8>, tensor<1xi16>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<?x2xi8>

  // CHECK: %[[C0:.+]] = arith.constant 0
  // CHECK: %[[BATCH:.+]] = tensor.dim %[[ARG0]], %[[C0]]
  // CHECK: %[[INIT:.+]] = tensor.empty(%[[BATCH]]) : tensor<?x2xi8>
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP0]]], iterator_types = ["parallel", "parallel"]} ins(%[[ARG0]] : tensor<?x2xi8>) outs(%[[INIT]] : tensor<?x2xi8>)
  %1 = tosa.rescale %arg0, %multiplier, %shift, %input_zp, %output_zp {scale32 = false, rounding_mode = #tosa.rounding_mode<SINGLE_ROUND>, per_channel = false, input_unsigned = false, output_unsigned = true} : (tensor<?x2xi8>, tensor<1xi16>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<?x2xi8>

  return
}

// -----

// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @rescale_dyn
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]:
func.func @rescale_dyn(%arg0 : tensor<1x?x?x32xi32>) -> () {
  %input_zp = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  %output_zp = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
  // CHECK: %[[C1:.+]] = arith.constant 1
  // CHECK: %[[DIM1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
  // CHECK: %[[C2:.+]] = arith.constant 2
  // CHECK: %[[DIM2:.+]] = tensor.dim %[[ARG0]], %[[C2]]
  // CHECK: %[[INIT:.+]] = tensor.empty(%[[DIM1]], %[[DIM2]])
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[ARG0]] : tensor<1x?x?x32xi32>) outs(%[[INIT]] : tensor<1x?x?x32xi8>)
  %multiplier = "tosa.const"() {values = dense<1376784203> : tensor<1xi32> } : () -> tensor<1xi32>
  %shift = "tosa.const"() {values = dense<38> : tensor<1xi8> } : () -> tensor<1xi8>
  %0 = tosa.rescale %arg0, %multiplier, %shift, %input_zp, %output_zp {rounding_mode = #tosa.rounding_mode<DOUBLE_ROUND>, input_zp = 0 : i32, output_zp = 0 : i32, per_channel = false, scale32 = true, input_unsigned = false, output_unsigned = false} : (tensor<1x?x?x32xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x?x?x32xi8>
  return
}

// -----
// CHECK: #[[$MAP0:.*]] = affine_map<(d0) -> (d0)>

// CHECK-LABEL: @rescale_i8_unsigned_input_explicit
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]:
func.func @rescale_i8_unsigned_input_explicit(%arg0 : tensor<2xui8>) -> () {
  // CHECK: [[C0:%.+]] = arith.constant 19689
  // CHECK: [[C1:%.+]] = arith.constant 15
  // CHECK: [[INIT:%.+]] = tensor.empty()
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP0]]], iterator_types = ["parallel"]} ins(%[[ARG0]] : tensor<2xui8>) outs([[INIT]] : tensor<2xi8>)
  // CHECK: ^bb0([[IN:%.+]]: ui8, [[UNUSED:%.+]]: i8):
  // CHECK-DAG: [[C17:%.+]] = arith.constant 17
  // CHECK-DAG: [[C22:%.+]] = arith.constant 22
  // CHECK-DAG: [[IN_UTOI:%.+]] = builtin.unrealized_conversion_cast [[IN]] : ui8 to i8
  // CHECK-DAG: [[IN32:%.+]] = arith.extui [[IN_UTOI]]
  // CHECK-DAG: [[IN_ZEROED:%.+]] = arith.subi [[IN32]], [[C17]]
  // CHECK-DAG: [[SCALED:%.+]] = tosa.apply_scale [[IN_ZEROED]], [[C0]], [[C1]] {rounding_mode = #tosa.rounding_mode<SINGLE_ROUND>}
  // CHECK-DAG: [[SCALED_ZEROED:%.+]] = arith.addi [[SCALED]], [[C22]]
  // CHECK-DAG: [[CMIN:%.+]] = arith.constant -128
  // CHECK-DAG: [[CMAX:%.+]] = arith.constant 127
  // CHECK-DAG: [[LOWER:%.+]] = arith.maxsi [[CMIN]], [[SCALED_ZEROED]]
  // CHECK: [[BOUNDED:%.+]] = arith.minsi [[CMAX]], [[LOWER]]
  // CHECK: [[TRUNC:%.+]] = arith.trunci [[BOUNDED]]
  // CHECK: linalg.yield [[TRUNC]]
  %multiplier = "tosa.const"() {values = dense<19689> : tensor<1xi16> } : () -> tensor<1xi16>
  %shift = "tosa.const"() {values = dense<15> : tensor<1xi8> } : () -> tensor<1xi8>
  %input_zp = "tosa.const"() {values = dense<17> : tensor<1xi8>} : () -> tensor<1xi8>
  %output_zp = "tosa.const"() {values = dense<22> : tensor<1xi8>} : () -> tensor<1xi8>
  %0 = tosa.rescale %arg0, %multiplier, %shift, %input_zp, %output_zp {scale32 = false, rounding_mode = #tosa.rounding_mode<SINGLE_ROUND>, per_channel = false, input_unsigned = true, output_unsigned = false} : (tensor<2xui8>, tensor<1xi16>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<2xi8>

  return
}

// -----
// CHECK: #[[$MAP0:.*]] = affine_map<(d0) -> (d0)>

// CHECK-LABEL: @rescale_i8_unsigned_input_implicit
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]:
func.func @rescale_i8_unsigned_input_implicit(%arg0 : tensor<2xi8>) -> () {
  // CHECK: [[C0:%.+]] = arith.constant 19689
  // CHECK: [[C1:%.+]] = arith.constant 15
  // CHECK: [[INIT:%.+]] = tensor.empty()
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP0]]], iterator_types = ["parallel"]} ins(%[[ARG0]] : tensor<2xi8>) outs([[INIT]] : tensor<2xi8>)
  // CHECK: ^bb0([[IN:%.+]]: i8, [[UNUSED:%.+]]: i8):
  // CHECK-NOT: builtin.unrealized_conversion_cast [[IN]]
  // CHECK-DAG: [[C128:%.+]] = arith.constant 128
  // CHECK-DAG: [[C22:%.+]] = arith.constant 22
  // CHECK-DAG: [[IN32:%.+]] = arith.extui [[IN]]
  // CHECK-DAG: [[IN_ZEROED:%.+]] = arith.subi [[IN32]], [[C128]]
  // CHECK-DAG: [[SCALED:%.+]] = tosa.apply_scale [[IN_ZEROED]], [[C0]], [[C1]] {rounding_mode = #tosa.rounding_mode<SINGLE_ROUND>}
  // CHECK-DAG: [[SCALED_ZEROED:%.+]] = arith.addi [[SCALED]], [[C22]]
  // CHECK-DAG: [[CMIN:%.+]] = arith.constant -128
  // CHECK-DAG: [[CMAX:%.+]] = arith.constant 127
  // CHECK-DAG: [[LOWER:%.+]] = arith.maxsi [[CMIN]], [[SCALED_ZEROED]]
  // CHECK-DAG: [[BOUNDED:%.+]] = arith.minsi [[CMAX]], [[LOWER]]
  // CHECK-DAG: [[TRUNC:%.+]] = arith.trunci [[BOUNDED]]
  // CHECK: linalg.yield [[TRUNC]]
  %multiplier = "tosa.const"() {values = dense<19689> : tensor<1xi16> } : () -> tensor<1xi16>
  %shift = "tosa.const"() {values = dense<15> : tensor<1xi8> } : () -> tensor<1xi8>
  %input_zp = "tosa.const"() {values = dense<-128> : tensor<1xi8>} : () -> tensor<1xi8>
  %output_zp = "tosa.const"() {values = dense<22> : tensor<1xi8>} : () -> tensor<1xi8>
  %0 = tosa.rescale %arg0, %multiplier, %shift, %input_zp, %output_zp {scale32 = false, rounding_mode = #tosa.rounding_mode<SINGLE_ROUND>, per_channel = false, input_unsigned = true, output_unsigned = false} : (tensor<2xi8>, tensor<1xi16>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<2xi8>

  return
}

// -----
// CHECK: #[[$MAP0:.*]] = affine_map<(d0) -> (d0)>

// CHECK-LABEL: @rescale_i8_unsigned_input_output_explicit
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]:
func.func @rescale_i8_unsigned_input_output_explicit(%arg0 : tensor<2xui8>) -> () {
  // CHECK: [[C0:%.+]] = arith.constant 19689
  // CHECK: [[C1:%.+]] = arith.constant 15
  // CHECK: [[INIT:%.+]] = tensor.empty()
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP0]]], iterator_types = ["parallel"]} ins(%[[ARG0]] : tensor<2xui8>) outs([[INIT]] : tensor<2xui8>)
  // CHECK: ^bb0([[IN:%.+]]: ui8, [[UNUSED:%.+]]: ui8):
  // CHECK-DAG: [[C17:%.+]] = arith.constant 17
  // CHECK-DAG: [[C22:%.+]] = arith.constant 22
  // CHECK-DAG: [[IN_UTOI:%.+]] = builtin.unrealized_conversion_cast [[IN]] : ui8 to i8
  // CHECK-DAG: [[IN32:%.+]] = arith.extui [[IN_UTOI]]
  // CHECK-DAG: [[IN_ZEROED:%.+]] = arith.subi [[IN32]], [[C17]]
  // CHECK-DAG: [[SCALED:%.+]] = tosa.apply_scale [[IN_ZEROED]], [[C0]], [[C1]] {rounding_mode = #tosa.rounding_mode<SINGLE_ROUND>}
  // CHECK-DAG: [[SCALED_ZEROED:%.+]] = arith.addi [[SCALED]], [[C22]]
  // CHECK-DAG: [[CMIN:%.+]] = arith.constant -128
  // CHECK-DAG: [[CMAX:%.+]] = arith.constant 127
  // CHECK-DAG: [[LOWER:%.+]] = arith.maxsi [[CMIN]], [[SCALED_ZEROED]]
  // CHECK: [[BOUNDED:%.+]] = arith.minsi [[CMAX]], [[LOWER]]
  // CHECK: [[TRUNC:%.+]] = arith.trunci [[BOUNDED]]
  // CHECK: [[TRUNC_ITOU:%.+]] = builtin.unrealized_conversion_cast [[TRUNC]] : i8 to ui8
  // CHECK: linalg.yield [[TRUNC_ITOU]]
  %multiplier = "tosa.const"() {values = dense<19689> : tensor<1xi16> } : () -> tensor<1xi16>
  %shift = "tosa.const"() {values = dense<15> : tensor<1xi8> } : () -> tensor<1xi8>
  %input_zp = "tosa.const"() {values = dense<17> : tensor<1xi8>} : () -> tensor<1xi8>
  %output_zp = "tosa.const"() {values = dense<22> : tensor<1xi8>} : () -> tensor<1xi8>
  %0 = tosa.rescale %arg0, %multiplier, %shift, %input_zp, %output_zp {scale32 = false, rounding_mode = #tosa.rounding_mode<SINGLE_ROUND>, per_channel = false, input_unsigned = true, output_unsigned = false} : (tensor<2xui8>, tensor<1xi16>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<2xui8>

  return
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0) -> (d0)>

// CHECK-LABEL: @rescale_per_channel
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]:
func.func @rescale_per_channel(%arg0 : tensor<3xi8>) -> (tensor<3xi8>) {
  // CHECK: [[MULTIPLIERS:%.+]] = arith.constant dense<[42, 43, 0]>
  // CHECK: [[SHIFTS:%.+]] = arith.constant dense<[14, 15, 0]>
  // CHECK: [[INIT:%.+]] = tensor.empty()
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP0]], #[[$MAP0]], #[[$MAP0]]], iterator_types = ["parallel"]} ins(%[[ARG0]], [[MULTIPLIERS]], [[SHIFTS]] : tensor<3xi8>, tensor<3xi32>, tensor<3xi8>) outs([[INIT]] : tensor<3xi8>)
  // CHECK: ^bb0([[IN:%.+]]: i8, [[MULTIPLIER:%.+]]: i32, [[SHIFT:%.+]]: i8, [[UNUSED:%.+]]: i8):
  // CHECK: [[C243:%.+]] = arith.constant 43
  // CHECK: [[C252:%.+]] = arith.constant 52

  // CHECK-DAG: [[IN32:%.+]] = arith.extsi [[IN]]
  // CHECK-DAG: [[IN_ZEROED:%.+]] = arith.subi [[IN32]], [[C243]]
  // CHECK-DAG: [[SCALED:%.+]] = tosa.apply_scale [[IN_ZEROED]], [[MULTIPLIER]], [[SHIFT]] {rounding_mode = #tosa.rounding_mode<SINGLE_ROUND>}
  // CHECK-DAG: [[SCALED_ZEROED:%.+]] = arith.addi [[SCALED]], [[C252]]
  // CHECK-DAG: [[CMIN:%.+]] = arith.constant -128
  // CHECK-DAG: [[CMAX:%.+]] = arith.constant 127
  // CHECK-DAG: [[LOWER:%.+]] = arith.maxsi [[CMIN]], [[SCALED_ZEROED]]
  // CHECK-DAG: [[BOUNDED:%.+]] = arith.minsi [[CMAX]], [[LOWER]]
  // CHECK-DAG: [[TRUNC:%.+]] = arith.trunci [[BOUNDED]]
  // CHECK-DAG: linalg.yield [[TRUNC]]
  %multiplier = "tosa.const"() {values = dense<[42, 43, 44]> : tensor<3xi16>} : () -> tensor<3xi16>
  %shift = "tosa.const"() {values = dense<[14, 15, 64]> : tensor<3xi8>} : () -> tensor<3xi8>
  %input_zp = "tosa.const"() {values = dense<43> : tensor<1xi8>} : () -> tensor<1xi8>
  %output_zp = "tosa.const"() {values = dense<52> : tensor<1xi8>} : () -> tensor<1xi8>
  %0 = tosa.rescale %arg0, %multiplier, %shift, %input_zp, %output_zp {scale32 = false, rounding_mode = #tosa.rounding_mode<SINGLE_ROUND>, per_channel = true, input_unsigned = false, output_unsigned = false} : (tensor<3xi8>, tensor<3xi16>, tensor<3xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<3xi8>

  // CHECK: return [[GENERIC]]
  return %0 : tensor<3xi8>
}

// -----

// CHECK-LABEL: @rescaleDoubleRound
func.func @rescaleDoubleRound(%arg0 : tensor<2xi8>) -> (tensor<2xi8>) {
  %multiplier = "tosa.const"() {values = dense<19689> : tensor<1xi32>} : () -> tensor<1xi32>
  %shift = "tosa.const"() {values = dense<33> : tensor<1xi8>} : () -> tensor<1xi8>
  %input_zp = "tosa.const"() {values = dense<43> : tensor<1xi8>} : () -> tensor<1xi8>
  %output_zp = "tosa.const"() {values = dense<52> : tensor<1xi8>} : () -> tensor<1xi8>

  // CHECK: linalg.generic
  // CHECK: tosa.apply_scale
  // CHECK-SAME: {rounding_mode = #tosa.rounding_mode<DOUBLE_ROUND>}
  %0 = tosa.rescale %arg0, %multiplier, %shift, %input_zp, %output_zp {scale32 = true, rounding_mode = #tosa.rounding_mode<DOUBLE_ROUND>, per_channel = false, input_unsigned = false, output_unsigned = false} : (tensor<2xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<2xi8>
  return %0 : tensor<2xi8>
}

// -----

// CHECK-LABEL: @rescaleUnnecessaryDoubleRound
func.func @rescaleUnnecessaryDoubleRound(%arg0 : tensor<2xi8>) -> (tensor<2xi8>) {
  %multiplier = "tosa.const"() {values = dense<19689> : tensor<1xi32>} : () -> tensor<1xi32>
  %shift = "tosa.const"() {values = dense<15> : tensor<1xi8>} : () -> tensor<1xi8>
  %input_zp = "tosa.const"() {values = dense<43> : tensor<1xi8>} : () -> tensor<1xi8>
  %output_zp = "tosa.const"() {values = dense<52> : tensor<1xi8>} : () -> tensor<1xi8>

  // CHECK: linalg.generic
  // CHECK: tosa.apply_scale
  // CHECK-SAME:  {rounding_mode = #tosa.rounding_mode<SINGLE_ROUND>}
  %0 = tosa.rescale %arg0, %multiplier, %shift, %input_zp, %output_zp {scale32 = true, rounding_mode = #tosa.rounding_mode<DOUBLE_ROUND>, per_channel = false, input_unsigned = false, output_unsigned = false} : (tensor<2xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<2xi8>
  return %0 : tensor<2xi8>
}

// -----

func.func @unsupportedRescaleInexactRound(%arg0 : tensor<2xi8>) -> (tensor<2xi8>) {
  %multiplier = "tosa.const"() {values = dense<19689> : tensor<1xi32> } : () -> tensor<1xi32>
  %shift = "tosa.const"() {values = dense<33> : tensor<1xi8> } : () -> tensor<1xi8>
  %input_zp = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
  %output_zp = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
  // expected-error@+1 {{failed to legalize operation 'tosa.rescale'}}
  %0 = tosa.rescale %arg0, %multiplier, %shift, %input_zp, %output_zp {input_zp = 243 : i32, output_zp = 252 : i32, scale32 = true, rounding_mode = #tosa.rounding_mode<INEXACT_ROUND>, per_channel = false, input_unsigned = false, output_unsigned = false} : (tensor<2xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<2xi8>
  return %0 : tensor<2xi8>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @reverse
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]:
func.func @reverse(%arg0: tensor<5x4xi32>) -> () {
  // CHECK: %[[C0:.+]] = arith.constant 0
  // CHECK: %[[RDIM:.+]] = tensor.dim %[[ARG0]], %[[C0]]
  // CHECK: %[[INIT:.+]] = tensor.empty()
  // CHECK: %[[GENERIC:.+]] = linalg.generic {indexing_maps = [#[[$MAP0]]], iterator_types = ["parallel", "parallel"]} outs(%[[INIT]] : tensor<5x4xi32>)
  // CHECK-DAG:   %[[I0:.+]] = linalg.index 0
  // CHECK-DAG:   %[[I1:.+]] = linalg.index 1
  // CHECK-DAG:   %[[SUB1:.+]] = arith.constant 1
  // CHECK-DAG:   %[[RDIM_MINUS_C1:.+]] = arith.subi %[[RDIM]], %[[SUB1]]
  // CHECK-DAG:   %[[READ_DIM:.+]] = arith.subi %[[RDIM_MINUS_C1]], %[[I0]]
  // CHECK-DAG:   %[[EXTRACT:.+]] = tensor.extract %arg0[%[[READ_DIM]], %[[I1]]] : tensor<5x4xi32>
  // CHECK:   linalg.yield %[[EXTRACT]]
  %0 = tosa.reverse %arg0 {axis = 0 : i32} : (tensor<5x4xi32>) -> tensor<5x4xi32>

  // CHECK: %[[C1:.+]] = arith.constant 1
  // CHECK: %[[RDIM:.+]] = tensor.dim %[[ARG0]], %[[C1]]
  // CHECK: %[[INIT:.+]] = tensor.empty()
  // CHECK: %[[GENERIC:.+]] = linalg.generic {indexing_maps = [#[[$MAP0]]], iterator_types = ["parallel", "parallel"]} outs(%[[INIT]] : tensor<5x4xi32>)
  // CHECK-DAG:   %[[I0:.+]] = linalg.index 0
  // CHECK-DAG:   %[[I1:.+]] = linalg.index 1
  // CHECK-DAG:   %[[SUB1:.+]] = arith.constant 1
  // CHECK-DAG:   %[[RDIM_MINUS_C1:.+]] = arith.subi %[[RDIM]], %[[SUB1]]
  // CHECK-DAG:   %[[READ_DIM:.+]] = arith.subi %[[RDIM_MINUS_C1]], %[[I1]]
  // CHECK-DAG:   %[[EXTRACT:.+]] = tensor.extract %arg0[%[[I0]], %[[READ_DIM]]] : tensor<5x4xi32>
  // CHECK:   linalg.yield %[[EXTRACT]]
  %1 = tosa.reverse %arg0 {axis = 1 : i32} : (tensor<5x4xi32>) -> tensor<5x4xi32>
  return
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0) -> (d0)>

// CHECK-LABEL: @reverse_dyn
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]:
func.func @reverse_dyn(%arg0: tensor<?xi32>) -> () {
  // CHECK: %[[C0_1:.+]] = arith.constant 0
  // CHECK: %[[D0_1:.+]] = tensor.dim %[[ARG0]], %[[C0_1]]
  // CHECK: %[[C0_2:.+]] = arith.constant 0
  // CHECK: %[[D0_2:.+]] = tensor.dim %[[ARG0]], %[[C0_2]]
  // CHECK: %[[INIT:.+]] = tensor.empty(%[[D0_1]])
  // CHECK: %[[GENERIC:.+]] = linalg.generic {indexing_maps = [#[[$MAP0]]], iterator_types = ["parallel"]} outs(%[[INIT]] : tensor<?xi32>)
  // CHECK-DAG:   %[[I0:.+]] = linalg.index 0
  // CHECK-DAG:   %[[SUB1:.+]] = arith.constant 1
  // CHECK-DAG:   %[[RDIM_MINUS_C1:.+]] = arith.subi %[[D0_2]], %[[SUB1]]
  // CHECK-DAG:   %[[READ_DIM:.+]] = arith.subi %[[RDIM_MINUS_C1]], %[[I0]]
  // CHECK-DAG:   %[[EXTRACT:.+]] = tensor.extract %arg0[%[[READ_DIM]]] : tensor<?xi32>
  // CHECK:   linalg.yield %[[EXTRACT]]
  %0 = tosa.reverse %arg0 {axis = 0 : i32} : (tensor<?xi32>) -> tensor<?xi32>
  return
}

// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @tile
// CHECK-SAME: %[[ARG0:.+]]: tensor<2x3xi8>
func.func @tile(%arg0 : tensor<2x3xi8>) -> () {
  // CHECK: [[INIT:%.+]] = tensor.empty()
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[ARG0]] : tensor<2x3xi8>) outs([[INIT]] : tensor<2x2x1x3xi8>)
  // CHECK: ^bb0(%[[ARG1:[0-9a-zA-Z_]+]]: i8
  // CHECK:   linalg.yield %[[ARG1]] : i8
  // CHECK: [[CONST3:%.+]] = tosa.const_shape {values = dense<[4, 3]> : tensor<2xindex>} : () -> !tosa.shape<2>
  // CHECK: tosa.reshape [[GENERIC]], [[CONST3]]
  %cst21 = tosa.const_shape { values = dense<[2, 1]> : tensor<2xindex> } : () -> !tosa.shape<2>
  %0 = tosa.tile %arg0, %cst21: (tensor<2x3xi8>, !tosa.shape<2>) -> tensor<4x3xi8>

  // CHECK: [[INIT:%.+]] = tensor.empty()
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[ARG0]] : tensor<2x3xi8>) outs([[INIT]] : tensor<1x2x2x3xi8>)
  // CHECK: ^bb0(%[[ARG1:[0-9a-zA-Z_]+]]: i8
  // CHECK:   linalg.yield %[[ARG1]] : i8
  // CHECK: [[CONST8:%.+]] = tosa.const_shape {values = dense<[2, 6]> : tensor<2xindex>} : () -> !tosa.shape<2>
  // tosa.reshape [[GENERIC]], [[CONST8]]
  %cst12 = tosa.const_shape { values = dense<[1, 2]> : tensor<2xindex> } : () -> !tosa.shape<2>
  %1 = tosa.tile %arg0, %cst12: (tensor<2x3xi8>, !tosa.shape<2>) -> tensor<2x6xi8>

  // CHECK: [[INIT:%.+]] = tensor.empty()
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[ARG0]] : tensor<2x3xi8>) outs([[INIT]] : tensor<5x2x7x3xi8>)
  // CHECK: ^bb0(%[[ARG1:[0-9a-zA-Z_]+]]: i8
  // CHECK:   linalg.yield %[[ARG1]] : i8
  %cst57 = tosa.const_shape { values = dense<[5, 7]> : tensor<2xindex> } : () -> !tosa.shape<2>
  // CHECK: [[CONST13:%.+]] = tosa.const_shape {values = dense<[10, 21]> : tensor<2xindex>} : () -> !tosa.shape<2>
  // CHECK: tosa.reshape [[GENERIC]], [[CONST13]]
  %2 = tosa.tile %arg0, %cst57: (tensor<2x3xi8>, !tosa.shape<2>)  -> tensor<10x21xi8>

  return
}

// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @tile_dyn_input
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]:
func.func @tile_dyn_input(%arg0 : tensor<?x3xi8>) -> () {
  // CHECK: %[[CST0:.+]] = arith.constant 0
  // CHECK: %[[DYN:.+]] = tensor.dim %[[ARG0]], %[[CST0]] : tensor<?x3xi8>
  // CHECK: %[[INIT:.+]] = tensor.empty(%[[DYN]])
  // CHECK: %[[GENERIC:.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[ARG0]] : tensor<?x3xi8>) outs(%[[INIT]] : tensor<2x?x1x3xi8>)
  // CHECK: ^bb0(%[[ARG1:.+]]: i8,
  // CHECK:   linalg.yield %[[ARG1]] : i8
  // CHECK: %[[CONST3:.+]] = tosa.const_shape {values = dense<[-1, 3]> : tensor<2xindex>} : () -> !tosa.shape<2>
  // CHECK: tosa.reshape %[[GENERIC]], %[[CONST3]]
  %cst21 = tosa.const_shape { values = dense<[2, 1]> : tensor<2xindex> } : () -> !tosa.shape<2>
  %0 = tosa.tile %arg0, %cst21: (tensor<?x3xi8>, !tosa.shape<2>)  -> tensor<?x3xi8>

  return
}

// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @tile_dyn_multiples
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]:
func.func @tile_dyn_multiples(%arg0 : tensor<2x3xi8>) -> () {
  // CHECK: %[[CST1:.+]] = arith.constant 1
  // CHECK: %[[DYN:.+]] = tensor.dim %[[ARG0]], %[[CST1]] : tensor<2x3xi8>
  // CHECK: %[[INIT:.+]] = tensor.empty(%[[DYN]])
  // CHECK: %[[GENERIC:.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[ARG0]] : tensor<2x3xi8>) outs(%[[INIT]] : tensor<2x2x?x3xi8>)
  // CHECK: ^bb0(%[[ARG1:.+]]: i8,
  // CHECK:   linalg.yield %[[ARG1]] : i8
  // CHECK: %[[CONST2:.+]] = tosa.const_shape {values = dense<[2, -1]> : tensor<2xindex>} : () -> !tosa.shape<2>
  // CHECK: tosa.reshape %[[GENERIC]], %[[CONST2]]
  %cst = tosa.const_shape { values = dense<[2, -1]> : tensor<2xindex> } : () -> !tosa.shape<2>
  %0 = tosa.tile %arg0, %cst: (tensor<2x3xi8>, !tosa.shape<2>)  -> tensor<2x?xi8>

  return
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1) -> (d1)>
// CHECK: #[[$MAP2:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK: #[[$MAP3:.*]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$MAP4:.*]] = affine_map<(d0) -> ()>

func.func @argmax(%arg0 : tensor<3x2xi32>, %arg1 : tensor<6xf32>) -> () {
  // CHECK: [[IDX_INIT:%.+]] = tensor.empty()
  // CHECK: [[IDX_MIN:%.+]] = arith.constant 0 : i32
  // CHECK: [[IDX_FILL:%.+]] = linalg.fill ins([[IDX_MIN]]{{.*}}outs([[IDX_INIT]]
  // CHECK: [[VAL_INIT:%.+]] = tensor.empty()
  // CHECK: [[VAL_MIN:%.+]] = arith.constant -2147483648
  // CHECK: [[VAL_FILL:%.+]] = linalg.fill ins([[VAL_MIN]]{{.*}}outs([[VAL_INIT]]
  // CHECK: linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP1]]], iterator_types = ["reduction", "parallel"]} ins(%[[ARG0]] : tensor<3x2xi32>) outs([[IDX_FILL]], [[VAL_FILL]] : tensor<2xi32>, tensor<2xi32>)
  // CHECK: ^bb0(%[[ARG1:[0-9a-zA-Z_]+]]: i32, %[[ARG2:[0-9a-zA-Z_]+]]: i32, %[[ARG3:[0-9a-zA-Z_]+]]: i32
  // CHECK:   [[IDX:%.+]] = linalg.index 0
  // CHECK:   [[CAST:%.+]] = arith.index_cast [[IDX]]
  // CHECK:   [[CMP:%.+]] = arith.cmpi sgt, %[[ARG1]], %[[ARG3]]
  // CHECK:   [[SELECT_VAL:%.+]] = arith.select [[CMP]], %[[ARG1]], %[[ARG3]]
  // CHECK:   [[SELECT_IDX:%.+]] = arith.select [[CMP]], [[CAST]], %[[ARG2]]
  // CHECK:   linalg.yield [[SELECT_IDX]], [[SELECT_VAL]]
  %0 = tosa.argmax %arg0 { axis = 0 : i32} : (tensor<3x2xi32>)  -> tensor<2xi32>

  // CHECK: [[IDX_INIT:%.+]] = tensor.empty()
  // CHECK: [[IDX_MIN:%.+]] = arith.constant 0 : i32
  // CHECK: [[IDX_FILL:%.+]] = linalg.fill ins([[IDX_MIN]]{{.*}}outs([[IDX_INIT]]
  // CHECK: [[VAL_INIT:%.+]] = tensor.empty()
  // CHECK: [[VAL_MIN:%.+]] = arith.constant -2147483648
  // CHECK: [[VAL_FILL:%.+]] = linalg.fill ins([[VAL_MIN]]{{.*}}outs([[VAL_INIT]]
  // CHECK: linalg.generic {indexing_maps = [#map, #map2, #map2], iterator_types = ["parallel", "reduction"]} ins(%[[ARG0]] : tensor<3x2xi32>) outs([[IDX_FILL]], [[VAL_FILL]] : tensor<3xi32>, tensor<3xi32>)
  // CHECK: ^bb0(%[[ARG1:[0-9a-zA-Z_]+]]: i32, %[[ARG2:[0-9a-zA-Z_]+]]: i32, %[[ARG3:[0-9a-zA-Z_]+]]: i32
  // CHECK:   [[IDX:%.+]] = linalg.index 1
  // CHECK:   [[CAST:%.+]] = arith.index_cast [[IDX]]
  // CHECK:   [[CMP:%.+]] = arith.cmpi sgt, %[[ARG1]], %[[ARG3]]
  // CHECK:   [[SELECT_VAL:%.+]] = arith.select [[CMP]], %[[ARG1]], %[[ARG3]]
  // CHECK:   [[SELECT_IDX:%.+]] = arith.select [[CMP]], [[CAST]], %[[ARG2]]
  // CHECK:   linalg.yield [[SELECT_IDX]], [[SELECT_VAL]]
  %1 = tosa.argmax %arg0 { axis = 1 : i32} : (tensor<3x2xi32>)  -> tensor<3xi32>

  // CHECK: arith.constant -3.40282347E+38 : f32
  // CHECK: linalg.index
  // CHECK: arith.index_cast
  // CHECK: arith.cmpf ugt
  // CHECK: arith.cmpf ord
  // CHECK: andi
  // CHECK: select
  // CHECK: select
  // CHECK: linalg.yield
  %2 = tosa.argmax %arg1 { axis = 0 : i32} : (tensor<6xf32>)  -> tensor<i32>

  return
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1) -> (d1)>

func.func @argmax_dyn_non_axis(%arg0 : tensor<3x?xi32>) -> () {
  // CHECK: %[[CST1:.+]] = arith.constant 1
  // CHECK: %[[DYN:.+]] = tensor.dim %[[ARG0]], %[[CST1]]
  // CHECK: %[[IDX_INIT:.+]] = tensor.empty(%[[DYN]])
  // CHECK: %[[IDX_MIN:.+]] = arith.constant 0 : i32
  // CHECK: %[[IDX_FILL:.+]] = linalg.fill ins(%[[IDX_MIN]]{{.*}}outs(%[[IDX_INIT]]
  // CHECK: %[[VAL_INIT:.+]] = tensor.empty(%[[DYN]])
  // CHECK: %[[VAL_MIN:.+]] = arith.constant -2147483648
  // CHECK: %[[VAL_FILL:.+]] = linalg.fill ins(%[[VAL_MIN]]{{.*}}outs(%[[VAL_INIT]]
  // CHECK: linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP1]]], iterator_types = ["reduction", "parallel"]} ins(%[[ARG0]] : tensor<3x?xi32>) outs(%[[IDX_FILL]], %[[VAL_FILL]] : tensor<?xi32>, tensor<?xi32>)
  // CHECK: ^bb0(%[[ARG1:[0-9a-zA-Z_]+]]: i32, %[[ARG2:[0-9a-zA-Z_]+]]: i32, %[[ARG3:[0-9a-zA-Z_]+]]: i32
  // CHECK:   %[[IDX:.+]] = linalg.index 0
  // CHECK:   %[[CAST:.+]] = arith.index_cast %[[IDX]]
  // CHECK:   %[[CMP:.+]] = arith.cmpi sgt, %[[ARG1]], %[[ARG3]]
  // CHECK:   %[[SELECT_VAL:.+]] = arith.select %[[CMP]], %[[ARG1]], %[[ARG3]]
  // CHECK:   %[[SELECT_IDX:.+]] = arith.select %[[CMP]], %[[CAST]], %[[ARG2]]
  // CHECK:   linalg.yield %[[SELECT_IDX]], %[[SELECT_VAL]]
  %0 = tosa.argmax %arg0 { axis = 0 : i32} : (tensor<3x?xi32>)  -> tensor<?xi32>
  return
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1) -> (d0)>

func.func @argmax_dyn_axis(%arg0 : tensor<3x?xi32>) -> () {
  // CHECK: %[[IDX_INIT:.+]] = tensor.empty()
  // CHECK: %[[IDX_MIN:.+]] = arith.constant 0 : i32
  // CHECK: %[[IDX_FILL:.+]] = linalg.fill ins(%[[IDX_MIN]]{{.*}}outs(%[[IDX_INIT]]
  // CHECK: %[[VAL_INIT:.+]] = tensor.empty()
  // CHECK: %[[VAL_MIN:.+]] = arith.constant -2147483648
  // CHECK: %[[VAL_FILL:.+]] = linalg.fill ins(%[[VAL_MIN]]{{.*}}outs(%[[VAL_INIT]]
  // CHECK: linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel", "reduction"]} ins(%[[ARG0]] : tensor<3x?xi32>) outs(%[[IDX_FILL]], %[[VAL_FILL]] : tensor<3xi32>, tensor<3xi32>)
  // CHECK:   %[[IDX:.+]] = linalg.index 1
  // CHECK:   %[[CAST:.+]] = arith.index_cast %[[IDX]]
  // CHECK:   %[[CMP:.+]] = arith.cmpi sgt, %[[ARG1]], %[[ARG3]]
  // CHECK:   %[[SELECT_VAL:.+]] = arith.select %[[CMP]], %[[ARG1]], %[[ARG3]]
  // CHECK:   %[[SELECT_IDX:.+]] = arith.select %[[CMP]], %[[CAST]], %[[ARG2]]
  // CHECK:   linalg.yield %[[SELECT_IDX]], %[[SELECT_VAL]]
  %0 = tosa.argmax %arg0 { axis = 1 : i32} : (tensor<3x?xi32>)  -> tensor<3xi32>
  return
}

// -----

// CHECK-LABEL: @gather_float
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]
// CHECK-SAME:  %[[ARG1:[0-9a-zA-Z_]*]]
func.func @gather_float(%arg0: tensor<2x3x2xf32>, %arg1: tensor<2x3xi32>) -> () {
  // CHECK: %[[INIT:.+]] = tensor.empty()
  // CHECK: %[[GENERIC:.+]] = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[ARG1]] : tensor<2x3xi32>) outs(%[[INIT]] : tensor<2x3x2xf32>)
  // CHECK: ^bb0(%[[BBARG0:.+]]: i32, %[[BBARG1:.+]]: f32)
  // CHECK:   %[[IDX0:.+]] = linalg.index 0
  // CHECK:   %[[CAST:.+]] = arith.index_cast %[[BBARG0]]
  // CHECK:   %[[IDX2:.+]] = linalg.index 2
  // CHECK:   %[[EXTRACT:.+]] = tensor.extract %[[ARG0]][%[[IDX0]], %[[CAST]], %[[IDX2]]] : tensor<2x3x2xf32>
  // CHECK:   linalg.yield %[[EXTRACT]]
  %0 = tosa.gather %arg0, %arg1 : (tensor<2x3x2xf32>, tensor<2x3xi32>)  -> tensor<2x3x2xf32>
  return
}

// -----

// CHECK-LABEL: @gather_float_dyn
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]
// CHECK-SAME:  %[[ARG1:[0-9a-zA-Z_]*]]
func.func @gather_float_dyn(%arg0: tensor<?x3x2xf32>, %arg1: tensor<?x3xi32>) -> () {
  // CHECK: %[[C0:.+]] = arith.constant 0
  // CHECK: %[[BATCH:.+]] = tensor.dim %[[ARG0]], %[[C0]]
  // CHECK: %[[INIT:.+]] = tensor.empty(%[[BATCH]])
  // CHECK: %[[GENERIC:.+]] = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[ARG1]] : tensor<?x3xi32>) outs(%[[INIT]] : tensor<?x3x2xf32>)
  // CHECK: ^bb0(%[[BBARG0:.+]]: i32, %[[BBARG1:.+]]: f32)
  // CHECK:   %[[IDX0:.+]] = linalg.index 0
  // CHECK:   %[[CAST:.+]] = arith.index_cast %[[BBARG0]]
  // CHECK:   %[[IDX2:.+]] = linalg.index 2
  // CHECK:   %[[EXTRACT:.+]] = tensor.extract %[[ARG0]][%[[IDX0]], %[[CAST]], %[[IDX2]]] : tensor<?x3x2xf32>
  // CHECK:   linalg.yield %[[EXTRACT]]
  %0 = tosa.gather %arg0, %arg1 : (tensor<?x3x2xf32>, tensor<?x3xi32>)  -> tensor<?x3x2xf32>
  return
}

// -----

// CHECK-LABEL: @gather_float_all_dynamic
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]
// CHECK-SAME:  %[[ARG1:[0-9a-zA-Z_]*]]
func.func @gather_float_all_dynamic(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?xi32>) -> () {
  // CHECK: %[[C0:.+]] = arith.constant 0
  // CHECK: %[[BATCH:.+]] = tensor.dim %[[ARG0]], %[[C0]]
  // CHECK: %[[C1:.+]] = arith.constant 1
  // CHECK: %[[INDEX:.+]] = tensor.dim %[[ARG1]], %[[C1]]
  // CHECK: %[[C2:.+]] = arith.constant 2
  // CHECK: %[[CHANNEL:.+]] = tensor.dim %[[ARG0]], %[[C2]]
  // CHECK: %[[INIT:.+]] = tensor.empty(%[[BATCH]], %[[INDEX]], %[[CHANNEL]])
  // CHECK: %[[GENERIC:.+]] = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[ARG1]] : tensor<?x?xi32>) outs(%[[INIT]] : tensor<?x?x?xf32>)
  // CHECK: ^bb0(%[[BBARG0:.+]]: i32, %[[BBARG1:.+]]: f32)
  // CHECK:   %[[IDX0:.+]] = linalg.index 0
  // CHECK:   %[[CAST:.+]] = arith.index_cast %[[BBARG0]]
  // CHECK:   %[[IDX2:.+]] = linalg.index 2
  // CHECK:   %[[EXTRACT:.+]] = tensor.extract %[[ARG0]][%[[IDX0]], %[[CAST]], %[[IDX2]]] : tensor<?x?x?xf32>
  // CHECK:   linalg.yield %[[EXTRACT]]
  %0 = tosa.gather %arg0, %arg1 : (tensor<?x?x?xf32>, tensor<?x?xi32>)  -> tensor<?x?x?xf32>
  return
}

// -----

// CHECK-LABEL: @gather_int
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]
// CHECK-SAME:  %[[ARG1:[0-9a-zA-Z_]*]]
func.func @gather_int(%arg0: tensor<2x3x2xi32>, %arg1: tensor<2x3xi32>) -> () {
  // CHECK: %[[INIT:.+]] = tensor.empty()
  // CHECK: %[[GENERIC:.+]] = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[ARG1]] : tensor<2x3xi32>) outs(%[[INIT]] : tensor<2x3x2xi32>)
  // CHECK: ^bb0(%[[BBARG0:.+]]: i32, %[[BBARG1:.+]]: i32)
  // CHECK:   %[[IDX0:.+]] = linalg.index 0
  // CHECK:   %[[CAST:.+]] = arith.index_cast %[[BBARG0]]
  // CHECK:   %[[IDX2:.+]] = linalg.index 2
  // CHECK:   %[[EXTRACT:.+]] = tensor.extract %[[ARG0]][%[[IDX0]], %[[CAST]], %[[IDX2]]] : tensor<2x3x2xi32>
  // CHECK:   linalg.yield %[[EXTRACT]]
  %0 = tosa.gather %arg0, %arg1 : (tensor<2x3x2xi32>, tensor<2x3xi32>)  -> tensor<2x3x2xi32>
  return
}

// -----

// CHECK-LABEL: @table8
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]:
// CHECK-SAME:  %[[ARG1:[0-9a-zA-Z_]*]]:
func.func @table8(%arg0: tensor<6xi8>, %arg1: tensor<512xi8>) -> () {
  // CHECK: %[[INIT:.+]] = tensor.empty()
  // CHECK: %[[GENERIC:.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%[[ARG0]] : tensor<6xi8>) outs(%[[INIT]] : tensor<6xi8>)
  // CHECK: ^bb0(%[[ARG_IN:.+]]: i8, %[[ARG_INIT:.+]]: i8)
  // CHECK:   %[[CAST:.+]] = arith.index_cast %[[ARG_IN]]
  // CHECK:   %[[OFFSET:.+]] = arith.constant 128
  // CHECK:   %[[ADD:.+]] = arith.addi %[[CAST]], %[[OFFSET]]
  // CHECK:   %[[EXTRACT:.+]] = tensor.extract %[[ARG1]][%[[ADD]]]
  // CHECK:   linalg.yield %[[EXTRACT]]
  %0 = tosa.table %arg0, %arg1 : (tensor<6xi8>, tensor<512xi8>)  -> tensor<6xi8>
  return
}

// -----

// CHECK-LABEL: @table16
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]:
// CHECK-SAME:  %[[ARG1:[0-9a-zA-Z_]*]]:
func.func @table16(%arg0: tensor<6xi16>, %arg1: tensor<513xi16>) -> () {
  // CHECK: %[[INIT:.+]] = tensor.empty()
  // CHECK: %[[GENERIC:.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%[[ARG0]] : tensor<6xi16>) outs(%[[INIT]] : tensor<6xi32>)
  // CHECK: ^bb0(%[[ARG2:.*]]: i16, %[[ARG3:.*]]: i32)
  // CHECK: %[[EXT_IN:.+]] = arith.extsi %[[ARG2]]
  // CHECK: %[[C32768:.+]] = arith.constant 32768
  // CHECK: %[[C7:.+]] = arith.constant 7
  // CHECK: %[[C1:.+]] = arith.constant 1
  // CHECK: %[[C127:.+]] = arith.constant 127
  // CHECK: %[[INADD:.+]] = arith.addi %[[EXT_IN]], %[[C32768]]
  // CHECK: %[[IDX:.+]] = arith.shrui %[[INADD]], %[[C7]]
  // CHECK: %[[FRACTION:.+]] = arith.andi %[[INADD]], %[[C127]]
  // CHECK: %[[IDXPLUS1:.+]] = arith.addi %[[IDX]], %[[C1]]
  // CHECK: %[[IDX_CAST:.+]] = arith.index_cast %[[IDX]]
  // CHECK: %[[IDXPLUS1_CAST:.+]] = arith.index_cast %[[IDXPLUS1]]
  // CHECK: %[[BASE:.+]] = tensor.extract %[[ARG1]][%[[IDX_CAST]]]
  // CHECK: %[[NEXT:.+]] = tensor.extract %[[ARG1]][%[[IDXPLUS1_CAST]]]
  // CHECK: %[[BASE_EXT:.+]] = arith.extsi %[[BASE]]
  // CHECK: %[[NEXT_EXT:.+]] = arith.extsi %[[NEXT]]
  // CHECK: %[[BASE_MUL:.+]] = arith.shli %[[BASE_EXT]], %[[C7]]
  // CHECK: %[[DIFF:.+]] = arith.subi %[[NEXT_EXT]], %[[BASE_EXT]]
  // CHECK: %[[DIFF_MUL:.+]] = arith.muli %[[DIFF]], %[[FRACTION]]
  // CHECK: %[[RESULT:.+]] = arith.addi %[[BASE_MUL]], %[[DIFF_MUL]]
  // CHECK: linalg.yield %[[RESULT]]
  %0 = tosa.table %arg0, %arg1 : (tensor<6xi16>, tensor<513xi16>)  -> tensor<6xi32>
  return
}

// -----

// CHECK-LABEL: @table8_dyn
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]:
// CHECK-SAME:  %[[ARG1:[0-9a-zA-Z_]*]]:
func.func @table8_dyn(%arg0: tensor<?xi8>, %arg1: tensor<512xi8>) -> () {
  // CHECK: %[[CST0:.+]] = arith.constant 0
  // CHECK: %[[DYN:.+]] = tensor.dim %[[ARG0]], %[[CST0]]
  // CHECK: %[[INIT:.+]] = tensor.empty(%[[DYN]])
  // CHECK: %[[GENERIC:.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%[[ARG0]] : tensor<?xi8>) outs(%[[INIT]] : tensor<?xi8>)
  // CHECK: ^bb0(%[[ARG_IN:.+]]: i8, %[[ARG_INIT:.+]]: i8)
  // CHECK:   %[[CAST:.+]] = arith.index_cast %[[ARG_IN]]
  // CHECK:   %[[OFFSET:.+]] = arith.constant 128
  // CHECK:   %[[ADD:.+]] = arith.addi %[[CAST]], %[[OFFSET]]
  // CHECK:   %[[EXTRACT:.+]] = tensor.extract %[[ARG1]][%[[ADD]]]
  // CHECK:   linalg.yield %[[EXTRACT]]
  %0 = tosa.table %arg0, %arg1 : (tensor<?xi8>, tensor<512xi8>)  -> tensor<?xi8>
  return
}

// -----

// CHECK-LABEL: @table8_dyn_table
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]:
// CHECK-SAME:  %[[ARG1:[0-9a-zA-Z_]*]]:
func.func @table8_dyn_table(%arg0: tensor<6xi8>, %arg1: tensor<?xi8>) -> () {
  // CHECK: %[[INIT:.+]] = tensor.empty()
  // CHECK: %[[GENERIC:.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%[[ARG0]] : tensor<6xi8>) outs(%[[INIT]] : tensor<6xi8>)
  // CHECK: ^bb0(%[[ARG_IN:.+]]: i8, %[[ARG_INIT:.+]]: i8)
  // CHECK:   %[[CAST:.+]] = arith.index_cast %[[ARG_IN]]
  // CHECK:   %[[OFFSET:.+]] = arith.constant 128
  // CHECK:   %[[ADD:.+]] = arith.addi %[[CAST]], %[[OFFSET]]
  // CHECK:   %[[EXTRACT:.+]] = tensor.extract %[[ARG1]][%[[ADD]]]
  // CHECK:   linalg.yield %[[EXTRACT]]
  %0 = tosa.table %arg0, %arg1 : (tensor<6xi8>, tensor<?xi8>)  -> tensor<6xi8>
  return
}

// -----
// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py
// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>

// CHECK-LABEL:   func.func @test_static_rfft2d(
// CHECK-SAME:                                  %[[VAL_0:.*]]: tensor<5x4x8xf32>) -> (tensor<5x4x5xf32>, tensor<5x4x5xf32>) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 8 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 4 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 5 : index
// CHECK:           %[[VAL_6:.*]] = tensor.empty() : tensor<5x4x5xf32>
// CHECK:           %[[VAL_7:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_8:.*]] = linalg.fill ins(%[[VAL_7]] : f32) outs(%[[VAL_6]] : tensor<5x4x5xf32>) -> tensor<5x4x5xf32>
// CHECK:           %[[VAL_9:.*]] = tensor.empty() : tensor<5x4x5xf32>
// CHECK:           %[[VAL_10:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_11:.*]] = linalg.fill ins(%[[VAL_10]] : f32) outs(%[[VAL_9]] : tensor<5x4x5xf32>) -> tensor<5x4x5xf32>
// CHECK:           %[[VAL_12:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_13:.*]] = arith.constant 4 : index
// CHECK:           %[[VAL_14:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_15:.*]] = arith.constant 8 : index
// CHECK:           %[[VAL_16:.*]] = arith.constant 6.28318548 : f32
// CHECK:           %[[VAL_17:.*]] = arith.index_castui %[[VAL_13]] : index to i32
// CHECK:           %[[VAL_18:.*]] = arith.uitofp %[[VAL_17]] : i32 to f32
// CHECK:           %[[VAL_19:.*]] = arith.index_castui %[[VAL_15]] : index to i32
// CHECK:           %[[VAL_20:.*]] = arith.uitofp %[[VAL_19]] : i32 to f32
// CHECK:           %[[VAL_21:.*]]:2 = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%[[VAL_0]] : tensor<5x4x8xf32>) outs(%[[VAL_8]], %[[VAL_11]] : tensor<5x4x5xf32>, tensor<5x4x5xf32>) {
// CHECK:           ^bb0(%[[VAL_22:.*]]: f32, %[[VAL_23:.*]]: f32, %[[VAL_24:.*]]: f32):
// CHECK:             %[[VAL_25:.*]] = linalg.index 1 : index
// CHECK:             %[[VAL_26:.*]] = linalg.index 2 : index
// CHECK:             %[[VAL_27:.*]] = linalg.index 3 : index
// CHECK:             %[[VAL_28:.*]] = linalg.index 4 : index
// CHECK:             %[[VAL_29:.*]] = index.mul %[[VAL_27]], %[[VAL_25]]
// CHECK:             %[[VAL_30:.*]] = index.mul %[[VAL_28]], %[[VAL_26]]
// CHECK:             %[[VAL_31:.*]] = index.remu %[[VAL_29]], %[[VAL_13]]
// CHECK:             %[[VAL_32:.*]] = index.remu %[[VAL_30]], %[[VAL_15]]
// CHECK:             %[[VAL_33:.*]] = arith.index_castui %[[VAL_31]] : index to i32
// CHECK:             %[[VAL_34:.*]] = arith.uitofp %[[VAL_33]] : i32 to f32
// CHECK:             %[[VAL_35:.*]] = arith.index_castui %[[VAL_32]] : index to i32
// CHECK:             %[[VAL_36:.*]] = arith.uitofp %[[VAL_35]] : i32 to f32
// CHECK:             %[[VAL_37:.*]] = arith.divf %[[VAL_34]], %[[VAL_18]] : f32
// CHECK:             %[[VAL_38:.*]] = arith.divf %[[VAL_36]], %[[VAL_20]] : f32
// CHECK:             %[[VAL_39:.*]] = arith.addf %[[VAL_37]], %[[VAL_38]] : f32
// CHECK:             %[[VAL_40:.*]] = arith.mulf %[[VAL_16]], %[[VAL_39]] : f32
// CHECK:             %[[VAL_41:.*]] = math.cos %[[VAL_40]] : f32
// CHECK:             %[[VAL_42:.*]] = math.sin %[[VAL_40]] : f32
// CHECK:             %[[VAL_43:.*]] = arith.mulf %[[VAL_22]], %[[VAL_41]] : f32
// CHECK:             %[[VAL_44:.*]] = arith.mulf %[[VAL_22]], %[[VAL_42]] : f32
// CHECK:             %[[VAL_45:.*]] = arith.addf %[[VAL_23]], %[[VAL_43]] : f32
// CHECK:             %[[VAL_46:.*]] = arith.subf %[[VAL_24]], %[[VAL_44]] : f32
// CHECK:             linalg.yield %[[VAL_45]], %[[VAL_46]] : f32, f32
// CHECK:           } -> (tensor<5x4x5xf32>, tensor<5x4x5xf32>)
// CHECK:           return %[[VAL_47:.*]]#0, %[[VAL_47]]#1 : tensor<5x4x5xf32>, tensor<5x4x5xf32>
// CHECK:         }
func.func @test_static_rfft2d(%arg0: tensor<5x4x8xf32>) -> (tensor<5x4x5xf32>, tensor<5x4x5xf32>) {
  %output_real, %output_imag = "tosa.rfft2d"(%arg0) {} : (tensor<5x4x8xf32>) -> (tensor<5x4x5xf32>, tensor<5x4x5xf32>)
  return %output_real, %output_imag : tensor<5x4x5xf32>, tensor<5x4x5xf32>
}

// -----
// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py
// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>

// CHECK-LABEL:   func.func @test_dynamic_rfft2d(
// CHECK-SAME:                                   %[[VAL_0:.*]]: tensor<?x?x?xf32>) -> (tensor<?x?x?xf32>, tensor<?x?x?xf32>) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_2:.*]] = tensor.dim %[[VAL_0]], %[[VAL_1]] : tensor<?x?x?xf32>
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = tensor.dim %[[VAL_0]], %[[VAL_3]] : tensor<?x?x?xf32>
// CHECK:           %[[VAL_5:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_6:.*]] = tensor.dim %[[VAL_0]], %[[VAL_5]] : tensor<?x?x?xf32>
// CHECK:           %[[VAL_7:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_8:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_9:.*]] = arith.divui %[[VAL_6]], %[[VAL_8]] : index
// CHECK:           %[[VAL_10:.*]] = arith.addi %[[VAL_9]], %[[VAL_7]] : index
// CHECK:           %[[VAL_11:.*]] = tensor.empty(%[[VAL_2]], %[[VAL_4]], %[[VAL_10]]) : tensor<?x?x?xf32>
// CHECK:           %[[VAL_12:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_13:.*]] = linalg.fill ins(%[[VAL_12]] : f32) outs(%[[VAL_11]] : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK:           %[[VAL_14:.*]] = tensor.empty(%[[VAL_2]], %[[VAL_4]], %[[VAL_10]]) : tensor<?x?x?xf32>
// CHECK:           %[[VAL_15:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_16:.*]] = linalg.fill ins(%[[VAL_15]] : f32) outs(%[[VAL_14]] : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK:           %[[VAL_17:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_18:.*]] = tensor.dim %[[VAL_0]], %[[VAL_17]] : tensor<?x?x?xf32>
// CHECK:           %[[VAL_19:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_20:.*]] = tensor.dim %[[VAL_0]], %[[VAL_19]] : tensor<?x?x?xf32>
// CHECK:           %[[VAL_21:.*]] = arith.constant 6.28318548 : f32
// CHECK:           %[[VAL_22:.*]] = arith.index_castui %[[VAL_18]] : index to i32
// CHECK:           %[[VAL_23:.*]] = arith.uitofp %[[VAL_22]] : i32 to f32
// CHECK:           %[[VAL_24:.*]] = arith.index_castui %[[VAL_20]] : index to i32
// CHECK:           %[[VAL_25:.*]] = arith.uitofp %[[VAL_24]] : i32 to f32
// CHECK:           %[[VAL_26:.*]]:2 = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%[[VAL_0]] : tensor<?x?x?xf32>) outs(%[[VAL_13]], %[[VAL_16]] : tensor<?x?x?xf32>, tensor<?x?x?xf32>) {
// CHECK:           ^bb0(%[[VAL_27:.*]]: f32, %[[VAL_28:.*]]: f32, %[[VAL_29:.*]]: f32):
// CHECK:             %[[VAL_30:.*]] = linalg.index 1 : index
// CHECK:             %[[VAL_31:.*]] = linalg.index 2 : index
// CHECK:             %[[VAL_32:.*]] = linalg.index 3 : index
// CHECK:             %[[VAL_33:.*]] = linalg.index 4 : index
// CHECK:             %[[VAL_34:.*]] = index.mul %[[VAL_32]], %[[VAL_30]]
// CHECK:             %[[VAL_35:.*]] = index.mul %[[VAL_33]], %[[VAL_31]]
// CHECK:             %[[VAL_36:.*]] = index.remu %[[VAL_34]], %[[VAL_18]]
// CHECK:             %[[VAL_37:.*]] = index.remu %[[VAL_35]], %[[VAL_20]]
// CHECK:             %[[VAL_38:.*]] = arith.index_castui %[[VAL_36]] : index to i32
// CHECK:             %[[VAL_39:.*]] = arith.uitofp %[[VAL_38]] : i32 to f32
// CHECK:             %[[VAL_40:.*]] = arith.index_castui %[[VAL_37]] : index to i32
// CHECK:             %[[VAL_41:.*]] = arith.uitofp %[[VAL_40]] : i32 to f32
// CHECK:             %[[VAL_42:.*]] = arith.divf %[[VAL_39]], %[[VAL_23]] : f32
// CHECK:             %[[VAL_43:.*]] = arith.divf %[[VAL_41]], %[[VAL_25]] : f32
// CHECK:             %[[VAL_44:.*]] = arith.addf %[[VAL_42]], %[[VAL_43]] : f32
// CHECK:             %[[VAL_45:.*]] = arith.mulf %[[VAL_21]], %[[VAL_44]] : f32
// CHECK:             %[[VAL_46:.*]] = math.cos %[[VAL_45]] : f32
// CHECK:             %[[VAL_47:.*]] = math.sin %[[VAL_45]] : f32
// CHECK:             %[[VAL_48:.*]] = arith.mulf %[[VAL_27]], %[[VAL_46]] : f32
// CHECK:             %[[VAL_49:.*]] = arith.mulf %[[VAL_27]], %[[VAL_47]] : f32
// CHECK:             %[[VAL_50:.*]] = arith.addf %[[VAL_28]], %[[VAL_48]] : f32
// CHECK:             %[[VAL_51:.*]] = arith.subf %[[VAL_29]], %[[VAL_49]] : f32
// CHECK:             linalg.yield %[[VAL_50]], %[[VAL_51]] : f32, f32
// CHECK:           } -> (tensor<?x?x?xf32>, tensor<?x?x?xf32>)
// CHECK:           return %[[VAL_52:.*]]#0, %[[VAL_52]]#1 : tensor<?x?x?xf32>, tensor<?x?x?xf32>
// CHECK:         }
func.func @test_dynamic_rfft2d(%arg0: tensor<?x?x?xf32>) -> (tensor<?x?x?xf32>, tensor<?x?x?xf32>) {
  %output_real, %output_imag = "tosa.rfft2d"(%arg0) {} : (tensor<?x?x?xf32>) -> (tensor<?x?x?xf32>, tensor<?x?x?xf32>)
  return %output_real, %output_imag : tensor<?x?x?xf32>, tensor<?x?x?xf32>
}

// -----
// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py
// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>

// CHECK-LABEL:   func.func @test_static_fft2d(
// CHECK-SAME:                                 %[[VAL_0:.*]]: tensor<8x8x8xf32>,
// CHECK-SAME:                                 %[[VAL_1:.*]]: tensor<8x8x8xf32>) -> (tensor<8x8x8xf32>, tensor<8x8x8xf32>) {
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<8x8x8xf32>
// CHECK:           %[[VAL_3:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_4:.*]] = linalg.fill ins(%[[VAL_3]] : f32) outs(%[[VAL_2]] : tensor<8x8x8xf32>) -> tensor<8x8x8xf32>
// CHECK:           %[[VAL_5:.*]] = tensor.empty() : tensor<8x8x8xf32>
// CHECK:           %[[VAL_6:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_7:.*]] = linalg.fill ins(%[[VAL_6]] : f32) outs(%[[VAL_5]] : tensor<8x8x8xf32>) -> tensor<8x8x8xf32>
// CHECK:           %[[VAL_8:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_9:.*]] = arith.constant 8 : index
// CHECK:           %[[VAL_10:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_11:.*]] = arith.constant 8 : index
// CHECK:           %[[VAL_12:.*]] = arith.constant 6.28318548 : f32
// CHECK:           %[[VAL_13:.*]] = arith.index_castui %[[VAL_9]] : index to i32
// CHECK:           %[[VAL_14:.*]] = arith.uitofp %[[VAL_13]] : i32 to f32
// CHECK:           %[[VAL_15:.*]] = arith.index_castui %[[VAL_11]] : index to i32
// CHECK:           %[[VAL_16:.*]] = arith.uitofp %[[VAL_15]] : i32 to f32
// CHECK:           %[[VAL_17:.*]]:2 = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%[[VAL_0]], %[[VAL_1]] : tensor<8x8x8xf32>, tensor<8x8x8xf32>) outs(%[[VAL_4]], %[[VAL_7]] : tensor<8x8x8xf32>, tensor<8x8x8xf32>) {
// CHECK:           ^bb0(%[[VAL_18:.*]]: f32, %[[VAL_19:.*]]: f32, %[[VAL_20:.*]]: f32, %[[VAL_21:.*]]: f32):
// CHECK:             %[[VAL_22:.*]] = linalg.index 1 : index
// CHECK:             %[[VAL_23:.*]] = linalg.index 2 : index
// CHECK:             %[[VAL_24:.*]] = linalg.index 3 : index
// CHECK:             %[[VAL_25:.*]] = linalg.index 4 : index
// CHECK:             %[[VAL_26:.*]] = index.mul %[[VAL_24]], %[[VAL_22]]
// CHECK:             %[[VAL_27:.*]] = index.mul %[[VAL_25]], %[[VAL_23]]
// CHECK:             %[[VAL_28:.*]] = index.remu %[[VAL_26]], %[[VAL_9]]
// CHECK:             %[[VAL_29:.*]] = index.remu %[[VAL_27]], %[[VAL_11]]
// CHECK:             %[[VAL_30:.*]] = arith.index_castui %[[VAL_28]] : index to i32
// CHECK:             %[[VAL_31:.*]] = arith.uitofp %[[VAL_30]] : i32 to f32
// CHECK:             %[[VAL_32:.*]] = arith.index_castui %[[VAL_29]] : index to i32
// CHECK:             %[[VAL_33:.*]] = arith.uitofp %[[VAL_32]] : i32 to f32
// CHECK:             %[[VAL_34:.*]] = arith.divf %[[VAL_31]], %[[VAL_14]] : f32
// CHECK:             %[[VAL_35:.*]] = arith.divf %[[VAL_33]], %[[VAL_16]] : f32
// CHECK:             %[[VAL_36:.*]] = arith.addf %[[VAL_34]], %[[VAL_35]] : f32
// CHECK:             %[[VAL_37:.*]] = arith.mulf %[[VAL_12]], %[[VAL_36]] : f32
// CHECK:             %[[VAL_38:.*]] = math.cos %[[VAL_37]] : f32
// CHECK:             %[[VAL_39:.*]] = math.sin %[[VAL_37]] : f32
// CHECK:             %[[VAL_40:.*]] = arith.mulf %[[VAL_18]], %[[VAL_38]] : f32
// CHECK:             %[[VAL_41:.*]] = arith.mulf %[[VAL_19]], %[[VAL_39]] : f32
// CHECK:             %[[VAL_42:.*]] = arith.addf %[[VAL_40]], %[[VAL_41]] : f32
// CHECK:             %[[VAL_43:.*]] = arith.mulf %[[VAL_19]], %[[VAL_38]] : f32
// CHECK:             %[[VAL_44:.*]] = arith.mulf %[[VAL_18]], %[[VAL_39]] : f32
// CHECK:             %[[VAL_45:.*]] = arith.subf %[[VAL_43]], %[[VAL_44]] : f32
// CHECK:             %[[VAL_46:.*]] = arith.addf %[[VAL_20]], %[[VAL_42]] : f32
// CHECK:             %[[VAL_47:.*]] = arith.addf %[[VAL_21]], %[[VAL_45]] : f32
// CHECK:             linalg.yield %[[VAL_46]], %[[VAL_47]] : f32, f32
// CHECK:           } -> (tensor<8x8x8xf32>, tensor<8x8x8xf32>)
// CHECK:           return %[[VAL_48:.*]]#0, %[[VAL_48]]#1 : tensor<8x8x8xf32>, tensor<8x8x8xf32>
// CHECK:         }
func.func @test_static_fft2d(%arg0: tensor<8x8x8xf32>, %arg1: tensor<8x8x8xf32>) -> (tensor<8x8x8xf32>, tensor<8x8x8xf32>) {
  %output_real, %output_imag = "tosa.fft2d"(%arg0, %arg1) {inverse=false} : (tensor<8x8x8xf32>, tensor<8x8x8xf32>) -> (tensor<8x8x8xf32>, tensor<8x8x8xf32>)
  return %output_real, %output_imag : tensor<8x8x8xf32>, tensor<8x8x8xf32>
}

// -----
// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py
// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>

// CHECK-LABEL:   func.func @test_dynamic_fft2d(
// CHECK-SAME:                                  %[[VAL_0:.*]]: tensor<?x?x?xf32>,
// CHECK-SAME:                                  %[[VAL_1:.*]]: tensor<?x?x?xf32>) -> (tensor<?x?x?xf32>, tensor<?x?x?xf32>) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = tensor.dim %[[VAL_0]], %[[VAL_2]] : tensor<?x?x?xf32>
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = tensor.dim %[[VAL_0]], %[[VAL_4]] : tensor<?x?x?xf32>
// CHECK:           %[[VAL_6:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_7:.*]] = tensor.dim %[[VAL_0]], %[[VAL_6]] : tensor<?x?x?xf32>
// CHECK:           %[[VAL_8:.*]] = tensor.empty(%[[VAL_3]], %[[VAL_5]], %[[VAL_7]]) : tensor<?x?x?xf32>
// CHECK:           %[[VAL_9:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_10:.*]] = linalg.fill ins(%[[VAL_9]] : f32) outs(%[[VAL_8]] : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK:           %[[VAL_11:.*]] = tensor.empty(%[[VAL_3]], %[[VAL_5]], %[[VAL_7]]) : tensor<?x?x?xf32>
// CHECK:           %[[VAL_12:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_13:.*]] = linalg.fill ins(%[[VAL_12]] : f32) outs(%[[VAL_11]] : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK:           %[[VAL_14:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_15:.*]] = tensor.dim %[[VAL_0]], %[[VAL_14]] : tensor<?x?x?xf32>
// CHECK:           %[[VAL_16:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_17:.*]] = tensor.dim %[[VAL_0]], %[[VAL_16]] : tensor<?x?x?xf32>
// CHECK:           %[[VAL_18:.*]] = arith.constant 6.28318548 : f32
// CHECK:           %[[VAL_19:.*]] = arith.index_castui %[[VAL_15]] : index to i32
// CHECK:           %[[VAL_20:.*]] = arith.uitofp %[[VAL_19]] : i32 to f32
// CHECK:           %[[VAL_21:.*]] = arith.index_castui %[[VAL_17]] : index to i32
// CHECK:           %[[VAL_22:.*]] = arith.uitofp %[[VAL_21]] : i32 to f32
// CHECK:           %[[VAL_23:.*]]:2 = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%[[VAL_0]], %[[VAL_1]] : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%[[VAL_10]], %[[VAL_13]] : tensor<?x?x?xf32>, tensor<?x?x?xf32>) {
// CHECK:           ^bb0(%[[VAL_24:.*]]: f32, %[[VAL_25:.*]]: f32, %[[VAL_26:.*]]: f32, %[[VAL_27:.*]]: f32):
// CHECK:             %[[VAL_28:.*]] = linalg.index 1 : index
// CHECK:             %[[VAL_29:.*]] = linalg.index 2 : index
// CHECK:             %[[VAL_30:.*]] = linalg.index 3 : index
// CHECK:             %[[VAL_31:.*]] = linalg.index 4 : index
// CHECK:             %[[VAL_32:.*]] = index.mul %[[VAL_30]], %[[VAL_28]]
// CHECK:             %[[VAL_33:.*]] = index.mul %[[VAL_31]], %[[VAL_29]]
// CHECK:             %[[VAL_34:.*]] = index.remu %[[VAL_32]], %[[VAL_15]]
// CHECK:             %[[VAL_35:.*]] = index.remu %[[VAL_33]], %[[VAL_17]]
// CHECK:             %[[VAL_36:.*]] = arith.index_castui %[[VAL_34]] : index to i32
// CHECK:             %[[VAL_37:.*]] = arith.uitofp %[[VAL_36]] : i32 to f32
// CHECK:             %[[VAL_38:.*]] = arith.index_castui %[[VAL_35]] : index to i32
// CHECK:             %[[VAL_39:.*]] = arith.uitofp %[[VAL_38]] : i32 to f32
// CHECK:             %[[VAL_40:.*]] = arith.divf %[[VAL_37]], %[[VAL_20]] : f32
// CHECK:             %[[VAL_41:.*]] = arith.divf %[[VAL_39]], %[[VAL_22]] : f32
// CHECK:             %[[VAL_42:.*]] = arith.addf %[[VAL_40]], %[[VAL_41]] : f32
// CHECK:             %[[VAL_43:.*]] = arith.mulf %[[VAL_18]], %[[VAL_42]] : f32
// CHECK:             %[[VAL_44:.*]] = arith.constant -1.000000e+00 : f32
// CHECK:             %[[VAL_45:.*]] = arith.mulf %[[VAL_43]], %[[VAL_44]] : f32
// CHECK:             %[[VAL_46:.*]] = math.cos %[[VAL_45]] : f32
// CHECK:             %[[VAL_47:.*]] = math.sin %[[VAL_45]] : f32
// CHECK:             %[[VAL_48:.*]] = arith.mulf %[[VAL_24]], %[[VAL_46]] : f32
// CHECK:             %[[VAL_49:.*]] = arith.mulf %[[VAL_25]], %[[VAL_47]] : f32
// CHECK:             %[[VAL_50:.*]] = arith.addf %[[VAL_48]], %[[VAL_49]] : f32
// CHECK:             %[[VAL_51:.*]] = arith.mulf %[[VAL_25]], %[[VAL_46]] : f32
// CHECK:             %[[VAL_52:.*]] = arith.mulf %[[VAL_24]], %[[VAL_47]] : f32
// CHECK:             %[[VAL_53:.*]] = arith.subf %[[VAL_51]], %[[VAL_52]] : f32
// CHECK:             %[[VAL_54:.*]] = arith.addf %[[VAL_26]], %[[VAL_50]] : f32
// CHECK:             %[[VAL_55:.*]] = arith.addf %[[VAL_27]], %[[VAL_53]] : f32
// CHECK:             linalg.yield %[[VAL_54]], %[[VAL_55]] : f32, f32
// CHECK:           } -> (tensor<?x?x?xf32>, tensor<?x?x?xf32>)
// CHECK:           return %[[VAL_56:.*]]#0, %[[VAL_56]]#1 : tensor<?x?x?xf32>, tensor<?x?x?xf32>
// CHECK:         }
func.func @test_dynamic_fft2d(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) -> (tensor<?x?x?xf32>, tensor<?x?x?xf32>) {
  %output_real, %output_imag = "tosa.fft2d"(%arg0, %arg1) {inverse = true} : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> (tensor<?x?x?xf32>, tensor<?x?x?xf32>)
  return %output_real, %output_imag : tensor<?x?x?xf32>, tensor<?x?x?xf32>
}


// -----

// CHECK: #[[$MAP0:.+]] = affine_map<(d0) -> (d0)>

// CHECK-LABEL:   func.func @test_cast_fp32_i64(
// CHECK-SAME:                                  %[[ARG0:.*]]: tensor<1xf32>) -> tensor<1xi64> {
// CHECK:           %[[EMPTY_TENSOR:.*]] = tensor.empty() : tensor<1xi64>
// CHECK:           %[[RESULT:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP0]]], iterator_types = ["parallel"]} ins(%[[ARG0]] : tensor<1xf32>) outs(%[[EMPTY_TENSOR]] : tensor<1xi64>) {
// CHECK:           ^bb0(%[[IN:.*]]: f32, %[[OUT:.*]]: i64):
// CHECK:             %[[ROUND_EVEN:.*]] = math.roundeven %[[IN]] : f32
// CHECK:             %[[FP_INT_MIN:.*]] = arith.constant -9.22337203E+18 : f32
// CHECK:             %[[FP_INT_MAX_PLUS_ONE:.*]] = arith.constant 9.22337203E+18 : f32
// CHECK:             %[[INT_MAX:.*]] = arith.constant 9223372036854775807 : i64
// CHECK:             %[[MAX:.*]] = arith.maximumf %[[ROUND_EVEN]], %[[FP_INT_MIN]] : f32
// CHECK:             %[[FPTOSI:.*]] = arith.fptosi %[[MAX]] : f32 to i64
// CHECK:             %[[CMPF:.*]] = arith.cmpf uge, %[[ROUND_EVEN]], %[[FP_INT_MAX_PLUS_ONE]] : f32
// CHECK:             %[[SELECT:.*]] = arith.select %[[CMPF]], %[[INT_MAX]], %[[FPTOSI]] : i64
// CHECK:             linalg.yield %[[SELECT]] : i64
// CHECK:           } -> tensor<1xi64>
// CHECK:           return %[[RESULT]] : tensor<1xi64>
func.func @test_cast_fp32_i64(%arg0: tensor<1xf32>) -> (tensor<1xi64>) {
  %0 = tosa.cast %arg0 : (tensor<1xf32>) -> tensor<1xi64>
  return %0: tensor<1xi64>
}

// -----

// CHECK-LABEL: @reduce_min_nan_propagate
func.func @reduce_min_nan_propagate(%arg0: tensor<5x4xf32>, %arg1: tensor<5x4xf32>) -> () {
  // CHECK: linalg.reduce
  // CHECK: arith.minimumf
  // CHECK-NOT: arith.cmpf uno
  // CHECK-NOT: arith.select
  // CHECK: linalg.yield
  // CHECK-NOT: arith.constant 0x7FC00000
  // CHECK-NOT: tensor.empty()
  // CHECK-NOT: linalg.fill
  // CHECK-NOT: tensor.empty()
  // CHECK-NOT: select
  // CHECK: return
  %3 = tosa.reduce_min %arg0 {axis = 0 : i32, nan_mode = #tosa.nan_mode<PROPAGATE>} : (tensor<5x4xf32>) -> tensor<1x4xf32>
  return
}

// -----

// CHECK-LABEL: @reduce_max_nan_propagate
func.func @reduce_max_nan_propagate(%arg0: tensor<5x4xf32>, %arg1: tensor<5x4xf32>) -> () {
  // CHECK: linalg.reduce
  // CHECK: arith.maximumf
  // CHECK-NOT: arith.cmpf uno
  // CHECK-NOT: arith.select
  // CHECK: linalg.yield
  // CHECK-NOT: arith.constant 0x7FC00000
  // CHECK-NOT: tensor.empty()
  // CHECK-NOT: linalg.fill
  // CHECK-NOT: tensor.empty()
  // CHECK-NOT: select
  // CHECK: return
  %4 = tosa.reduce_max %arg0 {axis = 0 : i32, nan_mode = #tosa.nan_mode<PROPAGATE>} : (tensor<5x4xf32>) -> tensor<1x4xf32>
  return
}

// -----

// CHECK-LABEL: @reduce_min_nan_ignore_int
func.func @reduce_min_nan_ignore_int(%arg0: tensor<5x4xi8>, %arg1: tensor<5x4xi8>) -> () {
  // CHECK: linalg.reduce
  // CHECK: arith.minsi
  // CHECK-NOT: arith.cmpf uno
  // CHECK-NOT: arith.select
  // CHECK: linalg.yield
  // CHECK-NOT: arith.constant 0x7FC00000
  // CHECK-NOT: tensor.empty()
  // CHECK-NOT: linalg.fill
  // CHECK-NOT: tensor.empty()
  // CHECK-NOT: select
  // CHECK: return
  %5 = tosa.reduce_min %arg0 {axis = 0 : i32, nan_mode = #tosa.nan_mode<IGNORE>} : (tensor<5x4xi8>) -> tensor<1x4xi8>
  return
}

// -----

// CHECK-LABEL: @reduce_max_nan_ignore_int
func.func @reduce_max_nan_ignore_int(%arg0: tensor<5x4xi8>, %arg1: tensor<5x4xi8>) -> () {
  // CHECK: linalg.reduce
  // CHECK: arith.maxsi
  // CHECK-NOT: arith.cmpf uno
  // CHECK-NOT: arith.select
  // CHECK: linalg.yield
  // CHECK-NOT: arith.constant 0x7FC00000
  // CHECK-NOT: tensor.empty()
  // CHECK-NOT: linalg.fill
  // CHECK-NOT: tensor.empty()
  // CHECK-NOT: select
  // CHECK: return
  %6 = tosa.reduce_max %arg0 {axis = 0 : i32, nan_mode = #tosa.nan_mode<IGNORE>} : (tensor<5x4xi8>) -> tensor<1x4xi8>
  return
}

// -----

// CHECK-LABEL: @reduce_min_nan_ignore
func.func @reduce_min_nan_ignore(%arg0: tensor<5x4xf32>, %arg1: tensor<5x4xf32>) -> () {
  // CHECK: linalg.reduce
  // CHECK: arith.minimumf
  // CHECK: arith.cmpf uno
  // CHECK: arith.select
  // CHECK: linalg.yield
  // CHECK: arith.constant 0x7FC00000
  // CHECK: tensor.empty()
  // CHECK: linalg.fill
  // CHECK: tensor.empty()
  // CHECK: select
  %5 = tosa.reduce_min %arg0 {axis = 0 : i32, nan_mode = #tosa.nan_mode<IGNORE>} : (tensor<5x4xf32>) -> tensor<1x4xf32>
  return
}

// -----

// CHECK-LABEL: @reduce_max_nan_ignore
func.func @reduce_max_nan_ignore(%arg0: tensor<5x4xf32>, %arg1: tensor<5x4xf32>) -> () {
  // CHECK: linalg.reduce
  // CHECK: arith.maximumf
  // CHECK: arith.cmpf uno
  // CHECK: arith.select
  // CHECK: linalg.yield
  // CHECK: arith.constant 0x7FC00000
  // CHECK: tensor.empty()
  // CHECK: linalg.fill
  // CHECK: tensor.empty()
  // CHECK: select
  %6 = tosa.reduce_max %arg0 {axis = 0 : i32, nan_mode = #tosa.nan_mode<IGNORE>} : (tensor<5x4xf32>) -> tensor<1x4xf32>
  return
}

// -----

// CHECK-LABEL: @minimum_nan_propagate
func.func @minimum_nan_propagate(%arg0: tensor<5x4xf32>, %arg1: tensor<5x4xf32>) -> () {
  // CHECK: linalg.generic
  // CHECK: arith.minimumf
  // CHECK-NOT: arith.cmpf uno
  // CHECK-NOT: arith.select
  // CHECK: linalg.yield
  %7 = tosa.minimum %arg0, %arg1 {nan_mode = #tosa.nan_mode<PROPAGATE>} : (tensor<5x4xf32>, tensor<5x4xf32>) -> tensor<5x4xf32>
  return
}

// -----

// CHECK-LABEL: @maximum_nan_propagate
func.func @maximum_nan_propagate(%arg0: tensor<5x4xf32>, %arg1: tensor<5x4xf32>) -> () {
  // CHECK: linalg.generic
  // CHECK: arith.maximumf
  // CHECK-NOT: arith.cmpf uno
  // CHECK-NOT: arith.select
  // CHECK: linalg.yield
  %8 = tosa.maximum %arg0, %arg1 {nan_mode = #tosa.nan_mode<PROPAGATE>} : (tensor<5x4xf32>, tensor<5x4xf32>) -> tensor<5x4xf32>
  return
}

// -----

// CHECK-LABEL: @minimum_nan_ignore_int
func.func @minimum_nan_ignore_int(%arg0: tensor<5x4xi8>, %arg1: tensor<5x4xi8>) -> () {
  // CHECK: linalg.generic
  // CHECK: arith.minsi
  // CHECK-NOT: arith.cmpf uno
  // CHECK-NOT: arith.select
  // CHECK: linalg.yield
  %9 = tosa.minimum %arg0, %arg1 {nan_mode = #tosa.nan_mode<IGNORE>} : (tensor<5x4xi8>, tensor<5x4xi8>) -> tensor<5x4xi8>
  return
}

// -----

// CHECK-LABEL: @maximum_nan_ignore_int
func.func @maximum_nan_ignore_int(%arg0: tensor<5x4xi8>, %arg1: tensor<5x4xi8>) -> () {
  // CHECK: linalg.generic
  // CHECK: arith.maxsi
  // CHECK-NOT: arith.cmpf uno
  // CHECK-NOT: arith.select
  // CHECK: linalg.yield
  %10 = tosa.maximum %arg0, %arg1 {nan_mode = #tosa.nan_mode<IGNORE>} : (tensor<5x4xi8>, tensor<5x4xi8>) -> tensor<5x4xi8>
  return
}

// -----

// CHECK-LABEL: @minimum_nan_ignore
func.func @minimum_nan_ignore(%arg0: tensor<5x4xf32>, %arg1: tensor<5x4xf32>) -> () {
  // CHECK: linalg.generic
  // CHECK: arith.minimumf
  // CHECK: arith.cmpf uno
  // CHECK: arith.cmpf uno
  // CHECK: arith.select
  // CHECK: arith.select
  // CHECK: linalg.yield
  %9 = tosa.minimum %arg0, %arg1 {nan_mode = #tosa.nan_mode<IGNORE>} : (tensor<5x4xf32>, tensor<5x4xf32>) -> tensor<5x4xf32>
  return
}

// -----

// CHECK-LABEL: @maximum_nan_ignore
func.func @maximum_nan_ignore(%arg0: tensor<5x4xf32>, %arg1: tensor<5x4xf32>) -> () {
  // CHECK: linalg.generic
  // CHECK: arith.maximumf
  // CHECK: arith.cmpf uno
  // CHECK: arith.cmpf uno
  // CHECK: arith.select
  // CHECK: arith.select
  // CHECK: linalg.yield
  %10 = tosa.maximum %arg0, %arg1 {nan_mode = #tosa.nan_mode<IGNORE>} : (tensor<5x4xf32>, tensor<5x4xf32>) -> tensor<5x4xf32>
  return
}

// -----

// CHECK-LABEL: @argmax_nan_propagate
func.func @argmax_nan_propagate(%arg0: tensor<5x4xf32>, %arg1: tensor<5x4xf32>) -> () {
  // CHECK: linalg.generic
  // CHECK: arith.cmpf ugt
  // CHECK: arith.cmpf ord
  // CHECK: andi
  // CHECK: arith.select
  // CHECK: arith.select
  // CHECK-NOT: arith.cmpf uno
  // CHECK-NOT: arith.select
  // CHECK: linalg.yield
  %11 = tosa.argmax %arg0 {axis = 0 : i32, nan_mode = #tosa.nan_mode<PROPAGATE>} : (tensor<5x4xf32>)  -> tensor<4xi32>
  return
}

// -----

// CHECK-LABEL: @argmax_nan_ignore_int
func.func @argmax_nan_ignore_int(%arg0: tensor<5x4xi8>, %arg1: tensor<5x4xi8>) -> () {
  // CHECK: linalg.generic
  // CHECK: arith.cmpi sgt
  // CHECK: arith.select
  // CHECK: arith.select
  // CHECK-NOT: arith.cmpf uno
  // CHECK-NOT: arith.cmpf uno
  // CHECK-NOT: arith.select
  // CHECK-NOT: arith.select
  // CHECK: linalg.yield
 %12 = tosa.argmax %arg0 {axis = 0 : i32, nan_mode = #tosa.nan_mode<IGNORE>} : (tensor<5x4xi8>)  -> tensor<4xi32>
  return
}

// -----

// CHECK-LABEL: @argmax_nan_ignore
func.func @argmax_nan_ignore(%arg0: tensor<5x4xf32>, %arg1: tensor<5x4xf32>) -> () {
  // CHECK: linalg.generic
  // CHECK: arith.cmpf ogt
  // CHECK: arith.select
  // CHECK: arith.select
  // CHECK: linalg.yield
  %12 = tosa.argmax %arg0 {axis = 0 : i32, nan_mode = #tosa.nan_mode<IGNORE>} : (tensor<5x4xf32>)  -> tensor<4xi32>
  return
}

// -----

// CHECK-LABEL: @clamp_nan_propagate
func.func @clamp_nan_propagate(%arg0: tensor<5x4xf32>, %arg1: tensor<5x4xf32>) -> () {
  // CHECK: linalg.generic
  // CHECK: arith.minimumf
  // CHECK: arith.maximumf
  // CHECK-NOT: arith.cmpf uno
  // CHECK-NOT: arith.select
  // CHECK: linalg.yield
  %13 = tosa.clamp %arg0 {min_val =  1.0 : f32, max_val = 5.0 : f32, nan_mode = #tosa.nan_mode<PROPAGATE>} : (tensor<5x4xf32>) -> tensor<5x4xf32>
  return
}

// -----

// CHECK-LABEL: @clamp_nan_ignore_int
func.func @clamp_nan_ignore_int(%arg0: tensor<5x4xi8>, %arg1: tensor<5x4xi8>) -> () {
  // CHECK: linalg.generic
  // CHECK: arith.maxsi
  // CHECK: arith.minsi
  // CHECK-NOT: arith.cmpf uno
  // CHECK-NOT: arith.select
  // CHECK: linalg.yield
  %14 = tosa.clamp %arg0 {min_val = 1 : i8, max_val = 5 : i8, nan_mode = #tosa.nan_mode<IGNORE>} : (tensor<5x4xi8>) -> tensor<5x4xi8>
  return
}

// -----

// CHECK-LABEL: @clamp_nan_ignore
func.func @clamp_nan_ignore(%arg0: tensor<5x4xf32>, %arg1: tensor<5x4xf32>) -> () {
  // CHECK: linalg.generic
  // CHECK: arith.minimumf
  // CHECK: arith.maximumf
  // CHECK: arith.cmpf uno
  // CHECK: arith.select
  // CHECK: linalg.yield
  %14 = tosa.clamp %arg0 {min_val = 1.0 : f32, max_val = 5.0 : f32, nan_mode = #tosa.nan_mode<IGNORE>} : (tensor<5x4xf32>) -> tensor<5x4xf32>

  return
}

// -----

// CHECK-LABEL: @test_0d_input
func.func @test_0d_input(%arg0: tensor<i32>) -> () {
  // CHECK: linalg.generic
  // CHECK: arith.muli
  %shift1 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %0 = tosa.mul %arg0, %arg0, %shift1 : (tensor<i32>, tensor<i32>, tensor<1xi8>) -> tensor<i32>

  // CHECK: linalg.generic
  // CHECK: ^bb0(%[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32):
  // CHECK: [[ZERO:%.+]] = arith.constant 0
  // CHECK: arith.subi [[ZERO]], %[[ARG1]]
  %in_zp = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
  %out_zp = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
  %5 = tosa.negate %arg0, %in_zp, %out_zp : (tensor<i32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>

  return
}

// -----

// CHECK-LABEL: @mul_no_const_shift
func.func @mul_no_const_shift(%arg0: tensor<2x3xi32>, %arg1: tensor<2x3xi32>, %arg2: tensor<1xi8>) -> tensor<2x3xi32> {
  // CHECK: linalg.generic
  // CHECK: ^bb0(%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i8, %[[OUT:.*]]: i32):
  // CHECK: tosa.apply_scale %[[ARG0]], %[[ARG1]], %[[ARG2]]
  %0 = tosa.mul %arg0, %arg1, %arg2 : (tensor<2x3xi32>, tensor<2x3xi32>, tensor<1xi8>) -> tensor<2x3xi32>
  return %0 : tensor<2x3xi32>
}
