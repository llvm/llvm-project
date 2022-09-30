// RUN: mlir-opt %s -test-linalg-transform-patterns=test-split-reduction  -split-input-file  | FileCheck %s
// RUN: mlir-opt %s -test-linalg-transform-patterns=test-split-reduction-inner-parallel  -split-input-file  | FileCheck %s --check-prefix=INNERPARALLELCHECK

func.func @matmul_split(%A : tensor<16x256xf32>, %B: tensor<256x32xf32>, %C: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = linalg.matmul ins(%A, %B: tensor<16x256xf32>, tensor<256x32xf32>)
                    outs(%C: tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0: tensor<16x32xf32>
}

//  CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
//  CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d2, d3, d1)>
//  CHECK-DAG: #[[$MAP2:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
//  CHECK-DAG: #[[$MAP3:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//  CHECK-DAG: #[[$MAP4:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//  CHECK-LABEL: @matmul_split
//  CHECK-DAG: %[[ID:.*]] = arith.constant 0.000000e+00 : f32
//  CHECK-DAG: %[[I1:.*]] = tensor.expand_shape %{{.*}}[0], [1, 2]] : tensor<16x256xf32> into tensor<16x4x64xf32>
//  CHECK-DAG: %[[I2:.*]] = tensor.expand_shape %{{.*}}[0, 1], [2]] : tensor<256x32xf32> into tensor<4x64x32xf32>
//  CHECK-DAG: %[[INI:.*]] = linalg.init_tensor [16, 32, 4] : tensor<16x32x4xf32>
//      CHECK: %[[F:.*]] = linalg.fill ins(%[[ID]] : f32) outs(%[[INI]] : tensor<16x32x4xf32>) -> tensor<16x32x4xf32>
//      CHECK: %[[G:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP2]]]
// CHECK-SAME:   , iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
// CHECK-SAME:   ins(%[[I1]], %[[I2]] : tensor<16x4x64xf32>, tensor<4x64x32xf32>) outs(%[[F]] : tensor<16x32x4xf32>) {
//      CHECK:   arith.mulf
//      CHECK:   arith.addf
//      CHECK:   linalg.yield
//      CHECK: } -> tensor<16x32x4xf32>
//      CHECK: %[[R:.*]] = linalg.generic {indexing_maps = [#[[$MAP3]], #[[$MAP4]]],
// CHECK-SAME:   iterator_types = ["parallel", "parallel", "reduction"]} ins(%[[G]] : tensor<16x32x4xf32>) outs(%{{.*}} : tensor<16x32xf32>) {
//      CHECK:   arith.addf
//      CHECK:   linalg.yield %{{.*}} : f32
//      CHECK: } -> tensor<16x32xf32>
//      CHECK: return %[[R]] : tensor<16x32xf32>

//  INNERPARALLELCHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
//  INNERPARALLELCHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d2, d3, d1)>
//  INNERPARALLELCHECK-DAG: #[[$MAP2:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
//  INNERPARALLELCHECK-DAG: #[[$MAP3:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//  INNERPARALLELCHECK-DAG: #[[$MAP4:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//  INNERPARALLELCHECK-LABEL: @matmul_split
//  INNERPARALLELCHECK-DAG: %[[ID:.*]] = arith.constant 0.000000e+00 : f32
//  INNERPARALLELCHECK-DAG: %[[I1:.*]] = tensor.expand_shape %{{.*}}[0], [1, 2]] : tensor<16x256xf32> into tensor<16x64x4xf32>
//  INNERPARALLELCHECK-DAG: %[[I2:.*]] = tensor.expand_shape %{{.*}}[0, 1], [2]] : tensor<256x32xf32> into tensor<64x4x32xf32>
//  INNERPARALLELCHECK-DAG: %[[INI:.*]] = linalg.init_tensor [16, 32, 4] : tensor<16x32x4xf32>
//      INNERPARALLELCHECK: %[[F:.*]] = linalg.fill ins(%[[ID]] : f32) outs(%[[INI]] : tensor<16x32x4xf32>) -> tensor<16x32x4xf32>
//      INNERPARALLELCHECK: %[[G:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP2]]]
// INNERPARALLELCHECK-SAME:   , iterator_types = ["parallel", "parallel", "reduction", "parallel"]}
// INNERPARALLELCHECK-SAME:   ins(%[[I1]], %[[I2]] : tensor<16x64x4xf32>, tensor<64x4x32xf32>) outs(%[[F]] : tensor<16x32x4xf32>) {
//      INNERPARALLELCHECK:   arith.mulf
//      INNERPARALLELCHECK:   arith.addf
//      INNERPARALLELCHECK:   linalg.yield
//      INNERPARALLELCHECK: } -> tensor<16x32x4xf32>
//      INNERPARALLELCHECK: %[[R:.*]] = linalg.generic {indexing_maps = [#[[$MAP3]], #[[$MAP4]]],
// INNERPARALLELCHECK-SAME:   iterator_types = ["parallel", "parallel", "reduction"]} ins(%[[G]] : tensor<16x32x4xf32>) outs(%{{.*}} : tensor<16x32xf32>) {
//      INNERPARALLELCHECK:   arith.addf
//      INNERPARALLELCHECK:   linalg.yield %{{.*}} : f32
//      INNERPARALLELCHECK: } -> tensor<16x32xf32>
//      INNERPARALLELCHECK: return %[[R]] : tensor<16x32xf32>

// -----

func.func @generic_split_1d(%arg0: tensor<32xf32>, %arg1: tensor<f32>, %out: tensor<f32>) -> tensor<f32> {
  %red = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>,
                                          affine_map<(d0) -> ()>,
                                          affine_map<(d0) -> ()>],
   iterator_types = ["reduction"]}
   ins(%arg0, %arg1 : tensor<32xf32>, tensor<f32>)
   outs(%out : tensor<f32>) {
    ^bb0(%arg7: f32, %arg8: f32, %arg9: f32):
      %40 = arith.subf %arg7, %arg8 : f32
      %41 = math.exp %40 : f32
      %42 = arith.mulf %41, %arg9 : f32
      linalg.yield %42 : f32
    } -> tensor<f32>
  return %red : tensor<f32>
}

//  CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
//  CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1) -> ()>
//  CHECK-DAG: #[[$MAP2:.*]] = affine_map<(d0, d1) -> (d0)>
//  CHECK-DAG: #[[$MAP3:.*]] = affine_map<(d0) -> (d0)>
//  CHECK-DAG: #[[$MAP4:.*]] = affine_map<(d0) -> ()>
//CHECK-LABEL: @generic_split_1d
//      CHECK: %[[ID:.*]] = arith.constant 1.000000e+00 : f32
//      CHECK: %[[I1:.*]] = tensor.expand_shape %{{.*}}[0, 1]] : tensor<32xf32> into tensor<4x8xf32>
//      CHECK: %[[INI:.*]] = linalg.init_tensor [4] : tensor<4xf32>
//      CHECK: %[[F:.*]] = linalg.fill ins(%[[ID]] : f32) outs(%[[INI]] : tensor<4xf32>) -> tensor<4xf32>
//      CHECK: %[[G:.*]] = linalg.generic
//      CHECK:   {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP2]]],
//      CHECK:   iterator_types = ["parallel", "reduction"]} ins(%[[I1]], %{{.*}} : tensor<4x8xf32>, tensor<f32>) outs(%[[F]] : tensor<4xf32>) {
//      CHECK:   arith.subf
//      CHECK:   math.exp
//      CHECK:   arith.mulf
//      CHECK:   linalg.yield
//      CHECK: } -> tensor<4xf32>
//      CHECK: %[[R:.*]] = linalg.generic {indexing_maps = [#[[$MAP3]], #[[$MAP4]]], iterator_types = ["reduction"]} ins(%[[G]] : tensor<4xf32>) outs(%{{.*}} : tensor<f32>) {
//      CHECK:   arith.mulf
//      CHECK:   linalg.yield
//      CHECK: } -> tensor<f32>
//      CHECK: return %[[R]] : tensor<f32>

//  INNERPARALLELCHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
//  INNERPARALLELCHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1) -> ()>
//  INNERPARALLELCHECK-DAG: #[[$MAP2:.*]] = affine_map<(d0, d1) -> (d1)>
//  INNERPARALLELCHECK-DAG: #[[$MAP3:.*]] = affine_map<(d0) -> (d0)>
//  INNERPARALLELCHECK-DAG: #[[$MAP4:.*]] = affine_map<(d0) -> ()>
//INNERPARALLELCHECK-LABEL: @generic_split_1d
//      INNERPARALLELCHECK: %[[ID:.*]] = arith.constant 1.000000e+00 : f32
//      INNERPARALLELCHECK: %[[I1:.*]] = tensor.expand_shape %{{.*}}[0, 1]] : tensor<32xf32> into tensor<8x4xf32>
//      INNERPARALLELCHECK: %[[INI:.*]] = linalg.init_tensor [4] : tensor<4xf32>
//      INNERPARALLELCHECK: %[[F:.*]] = linalg.fill ins(%[[ID]] : f32) outs(%[[INI]] : tensor<4xf32>) -> tensor<4xf32>
//      INNERPARALLELCHECK: %[[G:.*]] = linalg.generic
//      INNERPARALLELCHECK:   {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP2]]],
//      INNERPARALLELCHECK:   iterator_types = ["reduction", "parallel"]} ins(%[[I1]], %{{.*}} : tensor<8x4xf32>, tensor<f32>) outs(%[[F]] : tensor<4xf32>) {
//      INNERPARALLELCHECK:   arith.subf
//      INNERPARALLELCHECK:   math.exp
//      INNERPARALLELCHECK:   arith.mulf
//      INNERPARALLELCHECK:   linalg.yield
//      INNERPARALLELCHECK: } -> tensor<4xf32>
//      INNERPARALLELCHECK: %[[R:.*]] = linalg.generic {indexing_maps = [#[[$MAP3]], #[[$MAP4]]], iterator_types = ["reduction"]} ins(%[[G]] : tensor<4xf32>) outs(%{{.*}} : tensor<f32>) {
//      INNERPARALLELCHECK:   arith.mulf
//      INNERPARALLELCHECK:   linalg.yield
//      INNERPARALLELCHECK: } -> tensor<f32>
//      INNERPARALLELCHECK: return %[[R]] : tensor<f32>

// -----

func.func @generic_split_3d(%input: tensor<32x2xf32>, %input_2: tensor<5x32xf32>, %output: tensor<5x2xf32>)
  -> tensor<5x2xf32>
{
  %0 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d1, d0)>,
        affine_map<(d0, d1, d2) -> (d2, d1)>,
        affine_map<(d0, d1, d2) -> (d2, d0)>
      ],
      iterator_types = ["parallel", "reduction", "parallel"]
    } ins(%input, %input_2 : tensor<32x2xf32>, tensor<5x32xf32>) outs(%output : tensor<5x2xf32>) {
    ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
      %3 = arith.addf %arg0, %arg1 : f32
      %4 = arith.maxf %3, %arg2 : f32
      linalg.yield %4 : f32
    } -> tensor<5x2xf32>
  return %0 : tensor<5x2xf32>
}

//  CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0)>
//  CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d3, d2, d1)>
//  CHECK-DAG: #[[$MAP2:.*]] = affine_map<(d0, d1, d2, d3) -> (d3, d0, d2)>
//  CHECK-DAG: #[[$MAP3:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//  CHECK-DAG: #[[$MAP4:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL:  func @generic_split_3d
//      CHECK: %[[ID:.*]] = arith.constant -3.40282347E+38 : f32
//  CHECK-DAG: %[[I1:.*]] = tensor.expand_shape %{{.*}}[0, 1], [2]] : tensor<32x2xf32> into tensor<4x8x2xf32>
//  CHECK-DAG: %[[I2:.*]] = tensor.expand_shape %{{.*}}[0], [1, 2]] : tensor<5x32xf32> into tensor<5x4x8xf32>
//      CHECK: %[[INI:.*]] = linalg.init_tensor [5, 2, 4] : tensor<5x2x4xf32>
//      CHECK: %[[F:.*]] = linalg.fill ins(%[[ID]] : f32) outs(%[[INI]] : tensor<5x2x4xf32>) -> tensor<5x2x4xf32>
//      CHECK: %[[G:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP2]]], iterator_types = ["parallel", "reduction", "parallel", "parallel"]}
// CHECK-SAME:   ins(%[[I1]], %[[I2]] : tensor<4x8x2xf32>, tensor<5x4x8xf32>) outs(%[[F]] : tensor<5x2x4xf32>) {
//      CHECK:   arith.addf
//      CHECK:   arith.maxf
//      CHECK:   linalg.yield
//      CHECK: } -> tensor<5x2x4xf32>
//      CHECK: %[[R:.*]] = linalg.generic {indexing_maps = [#[[$MAP3]], #[[$MAP4]]], iterator_types = ["parallel", "parallel", "reduction"]}
// CHECK-SAME:   ins(%[[G]] : tensor<5x2x4xf32>) outs(%{{.*}} : tensor<5x2xf32>) {
//      CHECK:   arith.maxf
//      CHECK:   linalg.yield
//      CHECK:  } -> tensor<5x2xf32>
//      CHECK: return %[[R]] : tensor<5x2xf32>

//  INNERPARALLELCHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d1, d2, d0)>
//  INNERPARALLELCHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d3, d1, d2)>
//  INNERPARALLELCHECK-DAG: #[[$MAP2:.*]] = affine_map<(d0, d1, d2, d3) -> (d3, d0, d2)>
//  INNERPARALLELCHECK-DAG: #[[$MAP3:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//  INNERPARALLELCHECK-DAG: #[[$MAP4:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// INNERPARALLELCHECK-LABEL:  func @generic_split_3d
//      INNERPARALLELCHECK: %[[ID:.*]] = arith.constant -3.40282347E+38 : f32
//  INNERPARALLELCHECK-DAG: %[[I1:.*]] = tensor.expand_shape %{{.*}}[0, 1], [2]] : tensor<32x2xf32> into tensor<8x4x2xf32>
//  INNERPARALLELCHECK-DAG: %[[I2:.*]] = tensor.expand_shape %{{.*}}[0], [1, 2]] : tensor<5x32xf32> into tensor<5x8x4xf32>
//      INNERPARALLELCHECK: %[[INI:.*]] = linalg.init_tensor [5, 2, 4] : tensor<5x2x4xf32>
//      INNERPARALLELCHECK: %[[F:.*]] = linalg.fill ins(%[[ID]] : f32) outs(%[[INI]] : tensor<5x2x4xf32>) -> tensor<5x2x4xf32>
//      INNERPARALLELCHECK: %[[G:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP2]]], iterator_types = ["parallel", "reduction", "parallel", "parallel"]}
// INNERPARALLELCHECK-SAME:   ins(%[[I1]], %[[I2]] : tensor<8x4x2xf32>, tensor<5x8x4xf32>) outs(%[[F]] : tensor<5x2x4xf32>) {
//      INNERPARALLELCHECK:   arith.addf
//      INNERPARALLELCHECK:   arith.maxf
//      INNERPARALLELCHECK:   linalg.yield
//      INNERPARALLELCHECK: } -> tensor<5x2x4xf32>
//      INNERPARALLELCHECK: %[[R:.*]] = linalg.generic {indexing_maps = [#[[$MAP3]], #[[$MAP4]]], iterator_types = ["parallel", "parallel", "reduction"]}
// INNERPARALLELCHECK-SAME:   ins(%[[G]] : tensor<5x2x4xf32>) outs(%{{.*}} : tensor<5x2xf32>) {
//      INNERPARALLELCHECK:   arith.maxf
//      INNERPARALLELCHECK:   linalg.yield
//      INNERPARALLELCHECK:  } -> tensor<5x2xf32>
//      INNERPARALLELCHECK: return %[[R]] : tensor<5x2xf32>
