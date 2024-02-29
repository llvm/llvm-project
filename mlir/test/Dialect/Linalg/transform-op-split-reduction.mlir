// RUN: mlir-opt --split-input-file --transform-interpreter %s | FileCheck %s

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
//  CHECK-DAG: %[[INI:.*]] = tensor.empty() : tensor<16x32x4xf32>
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

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1:4 = transform.structured.split_reduction %0 { split_factor = 4, insert_split_dimension = 2}
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
  }
}

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
//  CHECK-DAG: %[[ID:.*]] = arith.constant 1.000000e+00 : f32
//  CHECK-DAG: %[[I1:.*]] = tensor.expand_shape %{{.*}}[0, 1]] : tensor<32xf32> into tensor<4x8xf32>
//  CHECK-DAG: %[[INI:.*]] = tensor.empty() : tensor<4xf32>
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

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1:4 = transform.structured.split_reduction %0 { split_factor = 4, insert_split_dimension = 0}
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
  }
}

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
      %4 = arith.maximumf %3, %arg2 : f32
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
//  CHECK-DAG: %[[ID:.*]] = arith.constant 0xFF800000 : f32
//  CHECK-DAG: %[[I1:.*]] = tensor.expand_shape %{{.*}}[0, 1], [2]] : tensor<32x2xf32> into tensor<4x8x2xf32>
//  CHECK-DAG: %[[I2:.*]] = tensor.expand_shape %{{.*}}[0], [1, 2]] : tensor<5x32xf32> into tensor<5x4x8xf32>
//  CHECK-DAG: %[[INI:.*]] = tensor.empty() : tensor<5x2x4xf32>
//      CHECK: %[[F:.*]] = linalg.fill ins(%[[ID]] : f32) outs(%[[INI]] : tensor<5x2x4xf32>) -> tensor<5x2x4xf32>
//      CHECK: %[[G:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP2]]], iterator_types = ["parallel", "reduction", "parallel", "parallel"]}
// CHECK-SAME:   ins(%[[I1]], %[[I2]] : tensor<4x8x2xf32>, tensor<5x4x8xf32>) outs(%[[F]] : tensor<5x2x4xf32>) {
//      CHECK:   arith.addf
//      CHECK:   arith.maximumf
//      CHECK:   linalg.yield
//      CHECK: } -> tensor<5x2x4xf32>
//      CHECK: %[[R:.*]] = linalg.generic {indexing_maps = [#[[$MAP3]], #[[$MAP4]]], iterator_types = ["parallel", "parallel", "reduction"]}
// CHECK-SAME:   ins(%[[G]] : tensor<5x2x4xf32>) outs(%{{.*}} : tensor<5x2xf32>) {
//      CHECK:   arith.maximumf
//      CHECK:   linalg.yield
//      CHECK:  } -> tensor<5x2xf32>
//      CHECK: return %[[R]] : tensor<5x2xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1:4 = transform.structured.split_reduction %0 { split_factor = 4, insert_split_dimension = 2}
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
  }
}

// -----

// Check that we don't use -inf as the neutral element for maxf when maxf has
// ninf. Instead check that we use the smallest finite floating point value.
// Also check that the fastmath flags are set on the created maxf
// instructions.
func.func @generic_split_3d_ninf(%input: tensor<32x2xf32>, %input_2: tensor<5x32xf32>, %output: tensor<5x2xf32>)
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
      %4 = arith.maximumf %3, %arg2 fastmath<nnan,ninf> : f32
      linalg.yield %4 : f32
    } -> tensor<5x2xf32>
  return %0 : tensor<5x2xf32>
}

//  CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0)>
//  CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d3, d2, d1)>
//  CHECK-DAG: #[[$MAP2:.*]] = affine_map<(d0, d1, d2, d3) -> (d3, d0, d2)>
//  CHECK-DAG: #[[$MAP3:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//  CHECK-DAG: #[[$MAP4:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL:  func @generic_split_3d_ninf
//  CHECK-DAG: %[[ID:.*]] = arith.constant -3.40282347E+38 : f32
//  CHECK-DAG: %[[I1:.*]] = tensor.expand_shape %{{.*}}[0, 1], [2]] : tensor<32x2xf32> into tensor<4x8x2xf32>
//  CHECK-DAG: %[[I2:.*]] = tensor.expand_shape %{{.*}}[0], [1, 2]] : tensor<5x32xf32> into tensor<5x4x8xf32>
//  CHECK-DAG: %[[INI:.*]] = tensor.empty() : tensor<5x2x4xf32>
//      CHECK: %[[F:.*]] = linalg.fill ins(%[[ID]] : f32) outs(%[[INI]] : tensor<5x2x4xf32>) -> tensor<5x2x4xf32>
//      CHECK: %[[G:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP2]]], iterator_types = ["parallel", "reduction", "parallel", "parallel"]}
// CHECK-SAME:   ins(%[[I1]], %[[I2]] : tensor<4x8x2xf32>, tensor<5x4x8xf32>) outs(%[[F]] : tensor<5x2x4xf32>) {
//      CHECK:   arith.addf
//      CHECK:   arith.maximumf {{.*}} fastmath<nnan,ninf>
//      CHECK:   linalg.yield
//      CHECK: } -> tensor<5x2x4xf32>
//      CHECK: %[[R:.*]] = linalg.generic {indexing_maps = [#[[$MAP3]], #[[$MAP4]]], iterator_types = ["parallel", "parallel", "reduction"]}
// CHECK-SAME:   ins(%[[G]] : tensor<5x2x4xf32>) outs(%{{.*}} : tensor<5x2xf32>) {
//      CHECK:   arith.maximumf {{.*}} fastmath<nnan,ninf>
//      CHECK:   linalg.yield
//      CHECK:  } -> tensor<5x2xf32>
//      CHECK: return %[[R]] : tensor<5x2xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1:4 = transform.structured.split_reduction %0 { split_factor = 4, insert_split_dimension = 2}
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
  }
}

// -----

func.func @matmul_split(%A : tensor<16x256xf32>, %B: tensor<256x32xf32>, %C: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = linalg.matmul ins(%A, %B: tensor<16x256xf32>, tensor<256x32xf32>)
                    outs(%C: tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0: tensor<16x32xf32>
}

//  CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
//  CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d2, d3, d1)>
//  CHECK-DAG: #[[$MAP2:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
//  CHECK-DAG: #[[$MAP3:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//  CHECK-DAG: #[[$MAP4:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//  CHECK-LABEL: @matmul_split
//  CHECK-DAG: %[[ID:.*]] = arith.constant 0.000000e+00 : f32
//  CHECK-DAG: %[[I1:.*]] = tensor.expand_shape %{{.*}}[0], [1, 2]] : tensor<16x256xf32> into tensor<16x64x4xf32>
//  CHECK-DAG: %[[I2:.*]] = tensor.expand_shape %{{.*}}[0, 1], [2]] : tensor<256x32xf32> into tensor<64x4x32xf32>
//  CHECK-DAG: %[[INI:.*]] = tensor.empty() : tensor<16x32x4xf32>
//      CHECK: %[[F:.*]] = linalg.fill ins(%[[ID]] : f32) outs(%[[INI]] : tensor<16x32x4xf32>) -> tensor<16x32x4xf32>
//      CHECK: %[[G:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP2]]]
// CHECK-SAME:   , iterator_types = ["parallel", "parallel", "reduction", "parallel"]}
// CHECK-SAME:   ins(%[[I1]], %[[I2]] : tensor<16x64x4xf32>, tensor<64x4x32xf32>) outs(%[[F]] : tensor<16x32x4xf32>) {
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

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1:4 = transform.structured.split_reduction %0 { split_factor = 4, insert_split_dimension = 2, inner_parallel}
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
  }
}

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
//  CHECK-DAG: #[[$MAP2:.*]] = affine_map<(d0, d1) -> (d1)>
//  CHECK-DAG: #[[$MAP3:.*]] = affine_map<(d0) -> (d0)>
//  CHECK-DAG: #[[$MAP4:.*]] = affine_map<(d0) -> ()>
//CHECK-LABEL: @generic_split_1d
//  CHECK-DAG: %[[ID:.*]] = arith.constant 1.000000e+00 : f32
//  CHECK-DAG: %[[I1:.*]] = tensor.expand_shape %{{.*}}[0, 1]] : tensor<32xf32> into tensor<8x4xf32>
//  CHECK-DAG: %[[INI:.*]] = tensor.empty() : tensor<4xf32>
//      CHECK: %[[F:.*]] = linalg.fill ins(%[[ID]] : f32) outs(%[[INI]] : tensor<4xf32>) -> tensor<4xf32>
//      CHECK: %[[G:.*]] = linalg.generic
//      CHECK:   {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP2]]],
//      CHECK:   iterator_types = ["reduction", "parallel"]} ins(%[[I1]], %{{.*}} : tensor<8x4xf32>, tensor<f32>) outs(%[[F]] : tensor<4xf32>) {
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

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1:4 = transform.structured.split_reduction %0 { split_factor = 4, insert_split_dimension = 0, inner_parallel}
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
  }
}

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
      %4 = arith.minimumf %3, %arg2 : f32
      linalg.yield %4 : f32
    } -> tensor<5x2xf32>
  return %0 : tensor<5x2xf32>
}

//  CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d1, d2, d0)>
//  CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d3, d1, d2)>
//  CHECK-DAG: #[[$MAP2:.*]] = affine_map<(d0, d1, d2, d3) -> (d3, d0, d2)>
//  CHECK-DAG: #[[$MAP3:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//  CHECK-DAG: #[[$MAP4:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL:  func @generic_split_3d
//  CHECK-DAG: %[[ID:.*]] = arith.constant 0x7F800000 : f32
//  CHECK-DAG: %[[I1:.*]] = tensor.expand_shape %{{.*}}[0, 1], [2]] : tensor<32x2xf32> into tensor<8x4x2xf32>
//  CHECK-DAG: %[[I2:.*]] = tensor.expand_shape %{{.*}}[0], [1, 2]] : tensor<5x32xf32> into tensor<5x8x4xf32>
//  CHECK-DAG: %[[INI:.*]] = tensor.empty() : tensor<5x2x4xf32>
//      CHECK: %[[F:.*]] = linalg.fill ins(%[[ID]] : f32) outs(%[[INI]] : tensor<5x2x4xf32>) -> tensor<5x2x4xf32>
//      CHECK: %[[G:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP2]]], iterator_types = ["parallel", "reduction", "parallel", "parallel"]}
// CHECK-SAME:   ins(%[[I1]], %[[I2]] : tensor<8x4x2xf32>, tensor<5x8x4xf32>) outs(%[[F]] : tensor<5x2x4xf32>) {
//      CHECK:   arith.addf
//      CHECK:   arith.minimumf
//      CHECK:   linalg.yield
//      CHECK: } -> tensor<5x2x4xf32>
//      CHECK: %[[R:.*]] = linalg.generic {indexing_maps = [#[[$MAP3]], #[[$MAP4]]], iterator_types = ["parallel", "parallel", "reduction"]}
// CHECK-SAME:   ins(%[[G]] : tensor<5x2x4xf32>) outs(%{{.*}} : tensor<5x2xf32>) {
//      CHECK:   arith.minimumf
//      CHECK:   linalg.yield
//      CHECK:  } -> tensor<5x2xf32>
//      CHECK: return %[[R]] : tensor<5x2xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1:4 = transform.structured.split_reduction %0 { split_factor = 4, insert_split_dimension = 2, inner_parallel}
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
  }
}

// -----

// Check that we don't use +inf as the neutral element for minf when minf has
// ninf. Instead check that we use the largest finite floating point value.
// Also check that the fastmath flags are set on the created minf
// instructions.
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
      %4 = arith.minimumf %3, %arg2 fastmath<ninf> : f32
      linalg.yield %4 : f32
    } -> tensor<5x2xf32>
  return %0 : tensor<5x2xf32>
}

//  CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d1, d2, d0)>
//  CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d3, d1, d2)>
//  CHECK-DAG: #[[$MAP2:.*]] = affine_map<(d0, d1, d2, d3) -> (d3, d0, d2)>
//  CHECK-DAG: #[[$MAP3:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//  CHECK-DAG: #[[$MAP4:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL:  func @generic_split_3d
//  CHECK-DAG: %[[ID:.*]] = arith.constant 3.40282347E+38 : f32
//  CHECK-DAG: %[[I1:.*]] = tensor.expand_shape %{{.*}}[0, 1], [2]] : tensor<32x2xf32> into tensor<8x4x2xf32>
//  CHECK-DAG: %[[I2:.*]] = tensor.expand_shape %{{.*}}[0], [1, 2]] : tensor<5x32xf32> into tensor<5x8x4xf32>
//  CHECK-DAG: %[[INI:.*]] = tensor.empty() : tensor<5x2x4xf32>
//      CHECK: %[[F:.*]] = linalg.fill ins(%[[ID]] : f32) outs(%[[INI]] : tensor<5x2x4xf32>) -> tensor<5x2x4xf32>
//      CHECK: %[[G:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP2]]], iterator_types = ["parallel", "reduction", "parallel", "parallel"]}
// CHECK-SAME:   ins(%[[I1]], %[[I2]] : tensor<8x4x2xf32>, tensor<5x8x4xf32>) outs(%[[F]] : tensor<5x2x4xf32>) {
//      CHECK:   arith.addf
//      CHECK:   arith.minimumf {{.*}} fastmath<ninf>
//      CHECK:   linalg.yield
//      CHECK: } -> tensor<5x2x4xf32>
//      CHECK: %[[R:.*]] = linalg.generic {indexing_maps = [#[[$MAP3]], #[[$MAP4]]], iterator_types = ["parallel", "parallel", "reduction"]}
// CHECK-SAME:   ins(%[[G]] : tensor<5x2x4xf32>) outs(%{{.*}} : tensor<5x2xf32>) {
//      CHECK:   arith.minimumf {{.*}} fastmath<ninf>
//      CHECK:   linalg.yield
//      CHECK:  } -> tensor<5x2xf32>
//      CHECK: return %[[R]] : tensor<5x2xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1:4 = transform.structured.split_reduction %0 { split_factor = 4, insert_split_dimension = 2, inner_parallel}
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
  }
}
