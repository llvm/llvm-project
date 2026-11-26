// RUN: mlir-opt -transform-interpreter -split-input-file -verify-diagnostics -allow-unregistered-dialect %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#reduction_2d_trait = {
  indexing_maps = [#map, #map1],
  iterator_types = ["parallel", "reduction"]
}

//    CHECK-DAG: #[[$PACKED_MAP_0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//    CHECK-DAG: #[[$PACKED_MAP_1:.*]] = affine_map<(d0, d1, d2) -> (d0)>

//  CHECK-LABEL: @reduction_2d_static
//   CHECK-SAME:   %[[T0:.+]]: tensor<3x7xf16>,
//   CHECK-SAME:   %[[T1:.+]]: tensor<3xf16>
func.func @reduction_2d_static(%t0: tensor<3x7xf16>, %t1: tensor<3xf16>) -> tensor<3xf16> {
  //      CHECK:  %[[EMPTY:.*]] = tensor.empty() : tensor<3x2x4xf16>
  //      CHECK: %[[PACKED:.*]] = linalg.pack %[[T0]] padding_value(%{{.*}} : f16)
  // CHECK-SAME:   inner_dims_pos = [1] inner_tiles = [4] into %[[EMPTY]] : tensor<3x7xf16> -> tensor<3x2x4xf16>
  //  CHECK-NOT: linalg.pack
  //      CHECK: linalg.generic
  // CHECK-SAME:   indexing_maps = [#[[$PACKED_MAP_0]], #[[$PACKED_MAP_1]]]
  // CHECK-SAME:   iterator_types = ["parallel", "reduction", "reduction"]
  // CHECK-SAME:   ins(%{{.*}} : tensor<3x2x4xf16>)
  // CHECK-SAME:  outs(%{{.*}} : tensor<3xf16>)
  %2 = linalg.generic #reduction_2d_trait ins(%t0 : tensor<3x7xf16>) outs(%t1 : tensor<3xf16>) {
  ^bb0(%in: f16, %out: f16):
    %3 = arith.addf %in, %out : f16
    linalg.yield %3 : f16
  } -> tensor<3xf16>

  //  CHECK-NOT: linalg.unpack
  return %2 : tensor<3xf16>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.pack %0 packed_sizes = [0, 4]
        : (!transform.any_op) -> (!transform.op<"linalg.generic">)
        transform.yield
  }
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#col_reduction_2d_trait = {
  indexing_maps = [#map, #map1],
  iterator_types = ["reduction", "parallel"]
}

//    CHECK-DAG: #[[$PACKED_MAP_0:.*]] = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
//    CHECK-DAG: #[[$PACKED_MAP_1:.*]] = affine_map<(d0, d1, d2) -> (d1)>

//  CHECK-LABEL: @col_reduction_2d_static
//   CHECK-SAME:   %[[T0:.+]]: tensor<7x3xf16>,
//   CHECK-SAME:   %[[T1:.+]]: tensor<3xf16>
func.func @col_reduction_2d_static(%t0: tensor<7x3xf16>, %t1: tensor<3xf16>) -> tensor<3xf16> {
  //      CHECK:  %[[EMPTY:.*]] = tensor.empty() : tensor<3x2x4xf16>
  //      CHECK: %[[PACKED:.*]] = linalg.pack %[[T0]] padding_value(%{{.*}} : f16)
  // CHECK-SAME:   outer_dims_perm = [1, 0] inner_dims_pos = [0] inner_tiles = [4] into %[[EMPTY]] : tensor<7x3xf16> -> tensor<3x2x4xf16>
  //  CHECK-NOT: linalg.pack
  //      CHECK: linalg.generic
  // CHECK-SAME:   indexing_maps = [#[[$PACKED_MAP_0]], #[[$PACKED_MAP_1]]]
  // CHECK-SAME:   iterator_types = ["reduction", "parallel", "reduction"]
  // CHECK-SAME:   ins(%{{.*}} : tensor<3x2x4xf16>)
  // CHECK-SAME:  outs(%{{.*}} : tensor<3xf16>)
  %2 = linalg.generic #col_reduction_2d_trait ins(%t0 : tensor<7x3xf16>) outs(%t1 : tensor<3xf16>) {
  ^bb0(%in: f16, %out: f16):
    %3 = arith.addf %in, %out : f16
    linalg.yield %3 : f16
  } -> tensor<3xf16>

  //  CHECK-NOT: linalg.unpack
  return %2 : tensor<3xf16>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.pack %0 packed_sizes = [4, 0]
        : (!transform.any_op) -> (!transform.op<"linalg.generic">)
    %pack = transform.get_producer_of_operand %1[0]
      : (!transform.op<"linalg.generic">) -> (!transform.op<"linalg.pack">)
    %2, %pack_2, %empty_unpack_2 =
      transform.structured.pack_transpose %pack with_compute_op(%1)
      outer_perm = [1, 0]
       : (!transform.op<"linalg.pack">, !transform.op<"linalg.generic">)
      -> (!transform.op<"linalg.generic">, !transform.op<"linalg.pack">, !transform.any_op)
      transform.yield
  }
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#reduction_2d_trait = {
  indexing_maps = [#map, #map1],
  iterator_types = ["parallel", "reduction"]
}

//    CHECK-DAG:     #[[$DIV4:.*]] = affine_map<()[s0] -> (s0 ceildiv 4)>
//    CHECK-DAG: #[[$PACKED_MAP_0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//    CHECK-DAG: #[[$PACKED_MAP_1:.*]] = affine_map<(d0, d1, d2) -> (d0)>

//  CHECK-LABEL: @reduction_2d_dynamic
//   CHECK-SAME:   %[[T0:.+]]: tensor<?x?xf16>,
//   CHECK-SAME:   %[[T1:.+]]: tensor<?xf16>
func.func @reduction_2d_dynamic(%t0: tensor<?x?xf16>, %t1: tensor<?xf16>) -> tensor<?xf16> {
  //  CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
  //  CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
  //  CHECK-DAG:     %[[D0:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<?x?xf16>
  //  CHECK-DAG:     %[[D1:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<?x?xf16>
  //      CHECK:   %[[D1B4:.*]] = affine.apply #[[$DIV4]]()[%[[D1]]]
  //      CHECK:  %[[EMPTY:.*]] = tensor.empty(%[[D0]], %[[D1B4]]) : tensor<?x?x4xf16>
  //      CHECK: %[[PACKED:.*]] = linalg.pack %[[T0]] padding_value(%{{.*}} : f16)
  // CHECK-SAME:   inner_dims_pos = [1] inner_tiles = [4] into %[[EMPTY]] : tensor<?x?xf16> -> tensor<?x?x4xf16>
  //  CHECK-NOT: linalg.pack
  //      CHECK: linalg.generic
  // CHECK-SAME:   indexing_maps = [#[[$PACKED_MAP_0]], #[[$PACKED_MAP_1]]]
  // CHECK-SAME:   iterator_types = ["parallel", "reduction", "reduction"]
  // CHECK-SAME:   ins(%{{.*}} : tensor<?x?x4xf16>)
  // CHECK-SAME:  outs(%{{.*}} : tensor<?xf16>)
  %2 = linalg.generic #reduction_2d_trait ins(%t0 : tensor<?x?xf16>) outs(%t1 : tensor<?xf16>) {
  ^bb0(%in: f16, %out: f16):
    %3 = arith.addf %in, %out : f16
    linalg.yield %3 : f16
  } -> tensor<?xf16>

  //  CHECK-NOT: linalg.unpack
  return %2 : tensor<?xf16>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.pack %0 packed_sizes = [0, 4]
        : (!transform.any_op) -> (!transform.op<"linalg.generic">)
        transform.yield
  }
}


// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#reduction_2d_trait = {
  indexing_maps = [#map, #map1],
  iterator_types = ["parallel", "reduction"]
}

//    CHECK-DAG:     #[[$DIV3:.*]] = affine_map<()[s0] -> (s0 ceildiv 3)>
//    CHECK-DAG:     #[[$DIV4:.*]] = affine_map<()[s0] -> (s0 ceildiv 4)>
//    CHECK-DAG: #[[$PACKED_MAP_0:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
//    CHECK-DAG: #[[$PACKED_MAP_1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d2)>

//  CHECK-LABEL: @reduction_2d_dynamic
//   CHECK-SAME:   %[[T0:.+]]: tensor<?x?xf16>,
//   CHECK-SAME:   %[[T1:.+]]: tensor<?xf16>
func.func @reduction_2d_dynamic(%t0: tensor<?x?xf16>, %t1: tensor<?xf16>) -> tensor<?xf16> {
  //      CHECK: %[[PACKED_0:.*]] = linalg.pack %[[T0]] padding_value(%{{.*}} : f16)
  // CHECK-SAME:   inner_dims_pos = [0, 1] inner_tiles = [3, 4] into %{{.*}} : tensor<?x?xf16> -> tensor<?x?x3x4xf16>
  //      CHECK: %[[PACKED_1:.*]] = linalg.pack %[[T1]] padding_value(%{{.*}} : f16)
  // CHECK-SAME:   inner_dims_pos = [0] inner_tiles = [3] into %{{.*}} : tensor<?xf16> -> tensor<?x3xf16>
  //  CHECK-NOT: linalg.pack
  //      CHECK: linalg.generic
  // CHECK-SAME:   indexing_maps = [#[[$PACKED_MAP_0]], #[[$PACKED_MAP_1]]]
  // CHECK-SAME:   iterator_types = ["parallel", "reduction", "parallel", "reduction"]
  // CHECK-SAME:   ins(%{{.*}} : tensor<?x?x3x4xf16>)
  // CHECK-SAME:  outs(%{{.*}} : tensor<?x3xf16>)
  %2 = linalg.generic #reduction_2d_trait ins(%t0 : tensor<?x?xf16>) outs(%t1 : tensor<?xf16>) {
  ^bb0(%in: f16, %out: f16):
    %3 = arith.addf %in, %out : f16
    linalg.yield %3 : f16
  } -> tensor<?xf16>

  //      CHECK: linalg.unpack %{{.*}} inner_dims_pos = [0] inner_tiles = [3] into %{{.*}} : tensor<?x3xf16> -> tensor<?xf16>
  return %2 : tensor<?xf16>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.pack %0 packed_sizes = [3, 4]
        : (!transform.any_op) -> (!transform.op<"linalg.generic">)
        transform.yield
  }
}

// -----

//                                                M   N   K   m   n   k       M   K   m   k
// CHECK-DAG: #[[$PACKED_MAP_0:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
//                                                                            K   N   n   k
// CHECK-DAG: #[[$PACKED_MAP_1:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d1, d4, d5)>
//                                                                            M   N   m   n
// CHECK-DAG: #[[$PACKED_MAP_2:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0, d4, d3)>

// CHECK-LABEL: @matmul
//  CHECK-SAME:   %[[A:[0-9a-zA-Z]+]]: tensor<?x?xf32>,
//  CHECK-SAME:   %[[B:[0-9a-zA-Z]+]]: tensor<?x?xf32>,
//  CHECK-SAME:   %[[C:[0-9a-zA-Z]+]]: tensor<?x?xf32>
func.func @matmul(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>)
    -> tensor<?x?xf32> {

  //      CHECK: %[[PACK_A:.*]] = linalg.pack %{{.*}} inner_dims_pos = [0, 1] inner_tiles = [2, 4]
  // CHECK-SAME:   : tensor<?x?xf32> -> tensor<?x?x2x4xf32>
  //      CHECK: %[[PACK_B:.*]] = linalg.pack %{{.*}} inner_dims_pos = [1, 0] inner_tiles = [3, 4]
  // CHECK-SAME:   : tensor<?x?xf32> -> tensor<?x?x3x4xf32>
  //      CHECK: %[[PACK_C:.*]] = linalg.pack %{{.*}} outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [3, 2]
  // CHECK-SAME:   : tensor<?x?xf32> -> tensor<?x?x3x2xf32>

  //      CHECK: linalg.generic {indexing_maps = [#[[$PACKED_MAP_0]], #[[$PACKED_MAP_1]], #[[$PACKED_MAP_2]]]
  // CHECK-SAME:     iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]}
  // CHECK-SAME:  ins(%{{.*}} : tensor<?x?x2x4xf32>, tensor<?x?x3x4xf32>)
  // CHECK-SAME: outs(%{{.*}} : tensor<?x?x3x2xf32>)
  %0 = linalg.matmul  ins(%A, %B: tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%C: tensor<?x?xf32>)
    -> tensor<?x?xf32>

  //      CHECK: linalg.unpack %{{.*}} outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [3, 2]
  // CHECK-SAME:   : tensor<?x?x3x2xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      //                                            M  N  K
      %1 = transform.structured.pack %0 packed_sizes = [2, 3, 4]
        : (!transform.any_op) -> (!transform.op<"linalg.generic">)

      %unpack = transform.get_consumers_of_result %1[0]
        : (!transform.op<"linalg.generic">) -> (!transform.op<"linalg.unpack">)
      %2, %pack_2, %unpack_2 =
        transform.structured.pack_transpose %unpack with_compute_op(%1)
        outer_perm = [1, 0] inner_perm = [1, 0]
        : (!transform.op<"linalg.unpack">, !transform.op<"linalg.generic">)
        -> (!transform.op<"linalg.generic">, !transform.op<"linalg.pack">, !transform.op<"linalg.unpack">)
        transform.yield
  }
}

// -----

//                                                N   F   H   W   C  KH  KW   f   c
// CHECK-DAG: #[[$PACKED_MAP_0:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d4, d2 + d5, d3 + d6, d8)>
// CHECK-DAG: #[[$PACKED_MAP_1:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d4, d5, d6, d7, d8)>
// CHECK-DAG: #[[$PACKED_MAP_2:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d7)>

// CHECK-LABEL: @conv_2d_nchw_fchw
//  CHECK-SAME:   %[[INPUT:.+]]: tensor<14x512x28x28xf32>,
//  CHECK-SAME:   %[[FILTER:.+]]: tensor<1024x512x1x1xf32>
//  CHECK-SAME:   %[[INIT:.+]]: tensor<14x1024x28x28xf32>
func.func @conv_2d_nchw_fchw(%i: tensor<14x512x28x28xf32>, %f: tensor<1024x512x1x1xf32>,
                             %o: tensor<14x1024x28x28xf32>) -> tensor<14x1024x28x28xf32> {

  //      CHECK: %[[PACK_INPUT:.*]] = linalg.pack %{{.*}} inner_dims_pos = [1] inner_tiles = [8]
  // CHECK-SAME:   : tensor<14x512x28x28xf32> -> tensor<14x64x28x28x8xf32>
  //      CHECK: %[[PACK_FILTER:.*]] = linalg.pack %{{.*}} inner_dims_pos = [0, 1] inner_tiles = [4, 8]
  // CHECK-SAME:   : tensor<1024x512x1x1xf32> -> tensor<256x64x1x1x4x8xf32>
  //      CHECK: %[[PACK_INPUT:.*]] = linalg.pack %{{.*}} inner_dims_pos = [1] inner_tiles = [4]
  // CHECK-SAME:   : tensor<14x1024x28x28xf32> -> tensor<14x256x28x28x4xf32>
  //      CHECK: linalg.generic {indexing_maps = [#[[$PACKED_MAP_0]], #[[$PACKED_MAP_1]], #[[$PACKED_MAP_2]]]
  // CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "parallel", "reduction"]}
  // CHECK-SAME:  ins(%{{.*}} : tensor<14x64x28x28x8xf32>, tensor<256x64x1x1x4x8xf32>)
  // CHECK-SAME: outs(%{{.*}} : tensor<14x256x28x28x4xf32>)
  %0 = linalg.conv_2d_nchw_fchw ins(%i, %f: tensor<14x512x28x28xf32>, tensor<1024x512x1x1xf32>)
                                outs(%o: tensor<14x1024x28x28xf32>) -> tensor<14x1024x28x28xf32>

  //      CHECK: linalg.unpack %{{.*}} inner_dims_pos = [1] inner_tiles = [4]
  // CHECK-SAME:   : tensor<14x256x28x28x4xf32> -> tensor<14x1024x28x28xf32>
  return %0: tensor<14x1024x28x28xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match interface{LinalgOp} in %arg1 : (!transform.any_op) -> !transform.any_op
    //                                            N  F  H  W  C KH KW
    %1 = transform.structured.pack %0 packed_sizes = [0, 4, 0, 0, 8, 0, 0]
        : (!transform.any_op) -> (!transform.op<"linalg.generic">)
        transform.yield
  }
}

// -----

//                                                N   H   W   F  KH  KW   C   f   c
// CHECK-DAG: #[[$PACKED_MAP_0:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1 + d4, d2 + d5, d6, d8)>
// CHECK-DAG: #[[$PACKED_MAP_1:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d4, d5, d6, d3, d7, d8)>
// CHECK-DAG: #[[$PACKED_MAP_2:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d7)>

// CHECK-LABEL: @conv_2d_nhwc_hwcf
//  CHECK-SAME:   %[[INPUT:.+]]: tensor<?x1x?x?xf32>,
//  CHECK-SAME:   %[[FILTER:.+]]: tensor<1x?x?x?xf32>
//  CHECK-SAME:   %[[INIT:.+]]: tensor<?x1x?x?xf32>
func.func @conv_2d_nhwc_hwcf(%input: tensor<?x1x?x?xf32>, %filter: tensor<1x?x?x?xf32>, %init: tensor<?x1x?x?xf32>) -> tensor<?x1x?x?xf32> {

  //      CHECK: %[[PACK_INPUT:.*]] = linalg.pack %{{.*}} inner_dims_pos = [3] inner_tiles = [6]
  // CHECK-SAME:   : tensor<?x1x?x?xf32> -> tensor<?x1x?x?x6xf32>
  //      CHECK: %[[PACK_FILTER:.*]] = linalg.pack %{{.*}} inner_dims_pos = [3, 2] inner_tiles = [4, 6]
  // CHECK-SAME:   : tensor<1x?x?x?xf32> -> tensor<1x?x?x?x4x6xf32>
  //      CHECK: %[[PACK_OUTPUT:.*]] = linalg.pack %{{.*}} inner_dims_pos = [3] inner_tiles = [4]
  // CHECK-SAME:   : tensor<?x1x?x?xf32> -> tensor<?x1x?x?x4xf32>

  //      CHECK: linalg.generic {indexing_maps = [#[[$PACKED_MAP_0]], #[[$PACKED_MAP_1]], #[[$PACKED_MAP_2]]]
  // CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "parallel", "reduction"]}
  // CHECK-SAME:  ins(%{{.*}} : tensor<?x1x?x?x6xf32>, tensor<1x?x?x?x4x6xf32>)
  // CHECK-SAME: outs(%{{.*}} : tensor<?x1x?x?x4xf32>)
  %0 = linalg.conv_2d_nhwc_hwcf
     ins (%input, %filter: tensor<?x1x?x?xf32>, tensor<1x?x?x?xf32>)
    outs (%init: tensor<?x1x?x?xf32>) -> tensor<?x1x?x?xf32>

  //      CHECK: linalg.unpack %{{.*}} inner_dims_pos = [3] inner_tiles = [4]
  // CHECK-SAME:   : tensor<?x1x?x?x4xf32> -> tensor<?x1x?x?xf32>
  return %0 : tensor<?x1x?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match interface{LinalgOp} in %arg1 : (!transform.any_op) -> !transform.any_op
    //                                            N  H  W  F KH KW  C
    %1 = transform.structured.pack %0 packed_sizes = [0, 0, 0, 4, 0, 0, 6]
        : (!transform.any_op) -> (!transform.op<"linalg.generic">)
        transform.yield
  }
}

// -----

// CHECK-DAG: affine_map<()[s0, s1] -> (s0 ceildiv s1)>
//                                                M   N   K    n   k      M   K   k
// CHECK-DAG: #[[$PACKED_MAP_0:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>
//                                                                        K   N   n   k
// CHECK-DAG: #[[$PACKED_MAP_1:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d2, d1, d3, d4)>
//                                                                        M   N    n
// CHECK-DAG: #[[$PACKED_MAP_2:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>

// CHECK-LABEL: @matmul_dynamic_pack_size
//  CHECK-SAME:   %[[A:[0-9a-zA-Z]+]]: tensor<?x?xf32>,
//  CHECK-SAME:   %[[B:[0-9a-zA-Z]+]]: tensor<?x?xf32>,
//  CHECK-SAME:   %[[C:[0-9a-zA-Z]+]]: tensor<?x?xf32>
func.func @matmul_dynamic_pack_size(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>)
    -> tensor<?x?xf32> {
  //      CHECK: %[[TS:.*]] = "some_tile_size"() : () -> index
  %sz = "some_tile_size"() : () -> (index)

  //      CHECK: %[[PACK_A:.*]] = linalg.pack %[[A]] {{.*}} inner_dims_pos = [1] inner_tiles = [%[[TS]]]
  // CHECK-SAME:   : tensor<?x?xf32> -> tensor<?x?x?xf32>
  //      CHECK: %[[PACK_B:.*]] = linalg.pack %[[B]] {{.*}} inner_dims_pos = [1, 0] inner_tiles = [%[[TS]], %[[TS]]]
  // CHECK-SAME:   : tensor<?x?xf32> -> tensor<?x?x?x?xf32>
  //      CHECK: %[[PACK_C:.*]] = linalg.pack %[[C]] {{.*}} inner_dims_pos = [1] inner_tiles = [%[[TS]]]
  // CHECK-SAME:   : tensor<?x?xf32> -> tensor<?x?x?xf32>
  //      CHECK: linalg.generic {indexing_maps = [#[[$PACKED_MAP_0]], #[[$PACKED_MAP_1]], #[[$PACKED_MAP_2]]]
  // CHECK-SAME:     iterator_types = ["parallel", "parallel", "reduction", "parallel", "reduction"]}
  // CHECK-SAME:  ins(%{{.*}} : tensor<?x?x?xf32>, tensor<?x?x?x?xf32>)
  // CHECK-SAME: outs(%{{.*}} : tensor<?x?x?xf32>)
  %0 = linalg.matmul  ins(%A, %B: tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%C: tensor<?x?xf32>)
    -> tensor<?x?xf32>

  //      CHECK: linalg.unpack %{{.*}} inner_dims_pos = [1] inner_tiles = [%[[TS]]] into %[[C]]
  // CHECK-SAME:   : tensor<?x?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      %sz = transform.structured.match ops{["some_tile_size"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      %1 = transform.structured.pack %0 packed_sizes = [0, %sz, %sz]
        : (!transform.any_op, !transform.any_op, !transform.any_op) -> (!transform.op<"linalg.generic">)
        transform.yield
  }
}

// -----

func.func @conv_cant_pack(%i: tensor<14x512x28x28xf32>, %f: tensor<1024x512x1x1xf32>,
                          %o: tensor<14x1024x28x28xf32>) -> tensor<14x1024x28x28xf32> {
  %0 = linalg.conv_2d_nchw_fchw ins(%i, %f: tensor<14x512x28x28xf32>, tensor<1024x512x1x1xf32>)
                                outs(%o: tensor<14x1024x28x28xf32>) -> tensor<14x1024x28x28xf32>
  return %0: tensor<14x1024x28x28xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match interface{LinalgOp} in %arg1 : (!transform.any_op) -> !transform.any_op
    //                                                N  F  H  W  C KH KW
    // expected-error @below {{data tiling failed}}
    %1 = transform.structured.pack %0 packed_sizes = [0, 0, 4, 0, 0, 0, 0]
        : (!transform.any_op) -> (!transform.op<"linalg.generic">)
        transform.yield
  }
}

// -----

func.func @matmul(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>)
    -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  %0 = linalg.matmul  ins(%A, %B: tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%C: tensor<?x?xf32>)
    -> tensor<?x?xf32>
  %1 = linalg.matmul  ins(%A, %B: tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%C: tensor<?x?xf32>)
    -> tensor<?x?xf32>
  return %0, %1 : tensor<?x?xf32>, tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      // expected-error @below {{requires target to map to exactly 1 LinalgOp (got 2)}}
      %1 = transform.structured.pack %0 packed_sizes = [2, 3, 4]
        : (!transform.any_op) -> (!transform.op<"linalg.generic">)
        transform.yield
  }
}


// -----

func.func @matmul(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>)
    -> tensor<?x?xf32> {
  %0 = linalg.matmul  ins(%A, %B: tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%C: tensor<?x?xf32>)
    -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      // expected-error @below {{requires number of packed sizes match the number of loops (2 vs 3)}}
      %1 = transform.structured.pack %0 packed_sizes = [2, 3]
        : (!transform.any_op) -> (!transform.op<"linalg.generic">)
        transform.yield
  }
}

// -----

func.func @matmul_unsupported_memref_pack(%A: memref<?x?xf32>,
                                          %B: memref<?x?xf32>,
                                          %C: memref<?x?xf32>) {
  linalg.matmul ins(%A, %B : memref<?x?xf32>, memref<?x?xf32>)
                outs(%C : memref<?x?xf32>)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      // expected-error @below {{structured.pack only supports tensor semantics. The target has memref operands}}
      %1 = transform.structured.pack %0 packed_sizes = [2, 3, 4]
        : (!transform.any_op) -> (!transform.op<"linalg.generic">)
        transform.yield
  }
}

// -----

func.func @pack_unsupported_memref_tile_using_for(%arg0: memref<128x256xf32>, %arg1: memref<4x8x32x32xf32>) {
  // expected-note @below {{target op}}
  linalg.pack %arg0 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %arg1
    : memref<128x256xf32> -> memref<4x8x32x32xf32>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.pack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      // expected-error @below {{tiling only supports tensor semantics for linalg.pack / linalg.unpack. The target has memref operands}}
      %1, %loops:2 = transform.structured.tile_using_for %0 tile_sizes [2, 4]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
        transform.yield
  }
}

// -----

func.func @pack_unsupported_memref_tile_using_forall(%arg0: memref<128x256xf32>, %arg1: memref<4x8x32x32xf32>) {
  // expected-note @below {{target op}}
  linalg.pack %arg0 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %arg1
    : memref<128x256xf32> -> memref<4x8x32x32xf32>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.pack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      // expected-error @below {{tiling only supports tensor semantics for linalg.pack / linalg.unpack. The target has memref operands}}
      %1:2 = transform.structured.tile_using_forall %0 num_threads [2, 4]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        transform.yield
  }
}

// -----

func.func @no_single_packing_op(%source: tensor<128x256xf32>, %dest: tensor<4x16x32x16xf32>) {
  %0 = linalg.pack %source inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %dest : tensor<128x256xf32> -> tensor<4x16x32x16xf32>
  %1 = linalg.unpack %0 inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %source : tensor<4x16x32x16xf32> -> tensor<128x256xf32>
  %2 = linalg.pack %source inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %dest : tensor<128x256xf32> -> tensor<4x16x32x16xf32>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.pack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      %1 = transform.structured.match ops{["linalg.unpack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        // expected-error @below {{requires target to map to exactly 1 packing op and 1 packed op (got 2 and 1)}}
      transform.structured.pack_transpose %0 with_compute_op(%1)
      inner_perm = [0]
        : (!transform.any_op, !transform.any_op)
        -> (!transform.any_op, !transform.any_op, !transform.any_op)
        transform.yield
  }
}

// -----

func.func @no_single_pack_unpack(%source: tensor<128x256xf32>, %dest: tensor<4x16x32x16xf32>) {
  %0 = arith.constant 0 : index
  %1 = tensor.empty() : tensor<f32>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["arith.constant"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      %1 = transform.structured.match ops{["tensor.empty"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        // expected-error @below {{requires target to map to a linalg.pack or linalg.unpack}}
      transform.structured.pack_transpose %0 with_compute_op(%1)
      inner_perm = [0]
        : (!transform.any_op, !transform.any_op)
        -> (!transform.any_op, !transform.any_op, !transform.any_op)
        transform.yield
  }
}

// -----

func.func @no_linalg_target(%source: tensor<128x256xf32>, %dest: tensor<4x16x32x16xf32>) {
  %0 = linalg.pack %source inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %dest : tensor<128x256xf32> -> tensor<4x16x32x16xf32>
  %1 = arith.constant 0 : index
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.pack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      %1 = transform.structured.match ops{["arith.constant"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        // expected-error @below {{requires a LinalgOp target}}
      transform.structured.pack_transpose %0 with_compute_op(%1)
      inner_perm = [0]
        : (!transform.any_op, !transform.any_op)
        -> (!transform.any_op, !transform.any_op, !transform.any_op)
        transform.yield
  }
}

// -----

func.func @no_single_use_by_linalg(%source: tensor<128x256xf32>, %dest: tensor<4x16x32x16xf32>) {
  %0 = linalg.pack %source inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %dest : tensor<128x256xf32> -> tensor<4x16x32x16xf32>
  %f0 = arith.constant 0.0 : f32
  %1 = tensor.empty() : tensor<f32>
  %2 = linalg.fill ins(%f0: f32) outs(%1 : tensor<f32>) -> tensor<f32>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.pack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      %1 = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        // expected-error @below {{not a single use by the LinalgOp target}}
      transform.structured.pack_transpose %0 with_compute_op(%1)
      inner_perm = [0]
        : (!transform.any_op, !transform.any_op)
        -> (!transform.any_op, !transform.any_op, !transform.any_op)
        transform.yield
  }
}

// -----

func.func @not_produced_by_linalg(%source: tensor<128x256xf32>, %dest: tensor<4x16x32x16xf32>) {
  %a = linalg.pack %source inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %dest : tensor<128x256xf32> -> tensor<4x16x32x16xf32>
  %b = linalg.unpack %a inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %source : tensor<4x16x32x16xf32> -> tensor<128x256xf32>
  %f0 = arith.constant 0.0 : f32
  %1 = tensor.empty() : tensor<f32>
  %2 = linalg.fill ins(%f0: f32) outs(%1 : tensor<f32>) -> tensor<f32>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.unpack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      %1 = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        // expected-error @below {{not produced by the LinalgOp target}}
      transform.structured.pack_transpose %0 with_compute_op(%1)
      inner_perm = [0]
        : (!transform.any_op, !transform.any_op)
        -> (!transform.any_op, !transform.any_op, !transform.any_op)
        transform.yield
  }
}

// -----

func.func @no_matching_pack(%source: tensor<16xf32>) {
  %f0 = arith.constant 0.0 : f32
  %1 = tensor.empty() : tensor<4x4xf32>
  %2 = linalg.fill ins(%f0: f32) outs(%1 : tensor<4x4xf32>) -> tensor<4x4xf32>
  %b = linalg.unpack %2 inner_dims_pos = [0] inner_tiles = [4] into %source : tensor<4x4xf32> -> tensor<16xf32>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.unpack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      %1 = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        // expected-error @below {{could not find matching pack op}}
      transform.structured.pack_transpose %0 with_compute_op(%1)
      inner_perm = [0]
        : (!transform.any_op, !transform.any_op)
        -> (!transform.any_op, !transform.any_op, !transform.any_op)
        transform.yield
  }
}

// -----

func.func @invalid_outer_perm(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>)
    -> tensor<?x?xf32> {
  %0 = linalg.matmul  ins(%A, %B: tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%C: tensor<?x?xf32>)
    -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      %1 = transform.structured.pack %0 packed_sizes = [2, 3, 4]
        : (!transform.any_op) -> (!transform.op<"linalg.generic">)

      %unpack = transform.get_consumers_of_result %1[0]
        : (!transform.op<"linalg.generic">) -> (!transform.op<"linalg.unpack">)
      %2, %pack_2, %unpack_2 =
        // expected-error @below {{"outer_perm" is not a valid permutation}}
        transform.structured.pack_transpose %unpack with_compute_op(%1)
        outer_perm = [1]
        : (!transform.op<"linalg.unpack">, !transform.op<"linalg.generic">)
        -> (!transform.op<"linalg.generic">, !transform.op<"linalg.pack">, !transform.op<"linalg.unpack">)
        transform.yield
  }
}

// -----

func.func @invalid_inner_perm(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>)
    -> tensor<?x?xf32> {
  %0 = linalg.matmul  ins(%A, %B: tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%C: tensor<?x?xf32>)
    -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      %1 = transform.structured.pack %0 packed_sizes = [2, 3, 4]
        : (!transform.any_op) -> (!transform.op<"linalg.generic">)

      %unpack = transform.get_consumers_of_result %1[0]
        : (!transform.op<"linalg.generic">) -> (!transform.op<"linalg.unpack">)
      %2, %pack_2, %unpack_2 =
        // expected-error @below {{"inner_perm" is not a valid permutation}}
        transform.structured.pack_transpose %unpack with_compute_op(%1)
        inner_perm = [1]
        : (!transform.op<"linalg.unpack">, !transform.op<"linalg.generic">)
        -> (!transform.op<"linalg.generic">, !transform.op<"linalg.pack">, !transform.op<"linalg.unpack">)
        transform.yield
  }
}

// -----

func.func @no_padding_on_packs(%A: tensor<32x32xf32>, %B: tensor<32x32xf32>, %C: tensor<32x32xf32>)
    -> tensor<32x32xf32> {
  %0 = linalg.matmul  ins(%A, %B: tensor<32x32xf32>, tensor<32x32xf32>)
                     outs(%C: tensor<32x32xf32>)
    -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}

// CHECK-LABEL: no_padding_on_packs
// CHECK: linalg.pack %{{.+}} inner_dims_pos = [0, 1] inner_tiles = [4, 8]
// CHECK-SAME:  into %{{.+}} : tensor<32x32xf32> -> tensor<8x4x4x8xf32>
// CHECK: linalg.pack %{{.+}} outer_dims_perm = [1, 0]
// CHECK-SAME:  inner_dims_pos = [0, 1] inner_tiles = [8, 8]
// CHECK-SAME:  into %{{.+}} : tensor<32x32xf32> -> tensor<4x4x8x8xf32>
// CHECK: linalg.pack %{{.+}} inner_dims_pos = [0, 1] inner_tiles = [4, 8]
// CHECK-SAME:  into %{{.+}} : tensor<32x32xf32> -> tensor<8x4x4x8xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      %1 = transform.structured.pack %0 packed_sizes = [4, 8, 8]
        : (!transform.any_op) -> (!transform.op<"linalg.generic">)
      %pack = transform.get_producer_of_operand %1[1]
      : (!transform.op<"linalg.generic">) -> (!transform.op<"linalg.pack">)
      %2, %pack_2, %empty_unpack_2 =
      transform.structured.pack_transpose %pack with_compute_op(%1)
      outer_perm = [1, 0] inner_perm = [1, 0]
       : (!transform.op<"linalg.pack">, !transform.op<"linalg.generic">)
      -> (!transform.op<"linalg.generic">, !transform.op<"linalg.pack">, !transform.any_op)
      transform.yield
  }
}

// -----

#identity = affine_map<(d0) -> (d0)>

// CHECK-DAG: #[[$PACK_INDEX_MAP:.*]] = affine_map<()[s0, s1] -> (s0 * 4 + s1)>
// CHECK-LABEL: func.func @pack_linalg_index
// CHECK:         linalg.generic
// CHECK:       ^bb0
// CHECK-DAG:     %[[OUTER:.*]] = linalg.index 0 : index
// CHECK-DAG:     %[[INNER:.*]] = linalg.index 1 : index
// CHECK:         %[[INDEX:.*]] = affine.apply #[[$PACK_INDEX_MAP]]()[%[[OUTER]], %[[INNER]]]
// CHECK:         arith.index_cast %[[INDEX]] : index to i32
func.func @pack_linalg_index(%input: tensor<8xi32>, %init: tensor<8xi32>) -> tensor<8xi32> {
  %result = linalg.generic {
      indexing_maps = [#identity, #identity],
      iterator_types = ["parallel"]}
      ins(%input : tensor<8xi32>) outs(%init : tensor<8xi32>) {
  ^bb0(%in: i32, %out: i32):
    %index = linalg.index 0 : index
    %index_i32 = arith.index_cast %index : index to i32
    %sum = arith.addi %in, %index_i32 : i32
    linalg.yield %sum : i32
  } -> tensor<8xi32>
  return %result : tensor<8xi32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %generic = transform.structured.match ops{["linalg.generic"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    transform.structured.pack %generic packed_sizes = [4]
      : (!transform.any_op) -> (!transform.op<"linalg.generic">)
    transform.yield
  }
}

// -----

#identity = affine_map<(d0) -> (d0)>

// Check index remapping when packing adds a padded iteration. Unpacking drops
// the extra output element at index 8.
// CHECK-DAG: #[[$PADDED_PACK_INDEX:.*]] = affine_map<()[s0, s1] -> (s0 * 3 + s1)>
// CHECK-LABEL: func.func @pack_linalg_index_padded(
// CHECK:         linalg.pack {{.*}} padding_value{{.*}} : tensor<8xi32> -> tensor<3x3xi32>
// CHECK:         %[[PACKED:.*]] = linalg.generic
// CHECK:       ^bb0
// CHECK-DAG:     %[[OUTER:.*]] = linalg.index 0 : index
// CHECK-DAG:     %[[INNER:.*]] = linalg.index 1 : index
// CHECK:         %[[INDEX:.*]] = affine.apply #[[$PADDED_PACK_INDEX]]()[%[[OUTER]], %[[INNER]]]
// CHECK:         arith.index_cast %[[INDEX]] : index to i32
// CHECK:         linalg.unpack %[[PACKED]] {{.*}} : tensor<3x3xi32> -> tensor<8xi32>
func.func @pack_linalg_index_padded(%input: tensor<8xi32>, %init: tensor<8xi32>) -> tensor<8xi32> {
  %result = linalg.generic {
      indexing_maps = [#identity, #identity],
      iterator_types = ["parallel"]}
      ins(%input : tensor<8xi32>) outs(%init : tensor<8xi32>) {
  ^bb0(%in: i32, %out: i32):
    %index = linalg.index 0 : index
    %index_i32 = arith.index_cast %index : index to i32
    %sum = arith.addi %in, %index_i32 : i32
    linalg.yield %sum : i32
  } -> tensor<8xi32>
  return %result : tensor<8xi32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %generic = transform.structured.match ops{["linalg.generic"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    transform.structured.pack %generic packed_sizes = [3]
      : (!transform.any_op) -> (!transform.op<"linalg.generic">)
    transform.yield
  }
}

// -----

#identity_3d = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-DAG: #[[$PACK_INDEX_4:.*]] = affine_map<()[s0, s1] -> (s0 * 4 + s1)>
// CHECK-DAG: #[[$PACK_INDEX_8:.*]] = affine_map<()[s0, s1] -> (s0 * 8 + s1)>
// CHECK-LABEL: func.func @pack_multiple_linalg_indices
// CHECK:         linalg.generic
// CHECK:       ^bb0
// CHECK:         %[[OUTER_0:.*]] = linalg.index 0 : index
// CHECK:         %[[INNER_0:.*]] = linalg.index 3 : index
// CHECK:         %[[INDEX_0:.*]] = affine.apply #[[$PACK_INDEX_4]]()[%[[OUTER_0]], %[[INNER_0]]]
// CHECK:         %[[INDEX_1:.*]] = linalg.index 1 : index
// CHECK:         %[[OUTER_2:.*]] = linalg.index 2 : index
// CHECK:         %[[INNER_2:.*]] = linalg.index 4 : index
// CHECK:         %[[INDEX_2:.*]] = affine.apply #[[$PACK_INDEX_8]]()[%[[OUTER_2]], %[[INNER_2]]]
// CHECK:         arith.index_cast %[[INDEX_0]] : index to i32
// CHECK:         arith.index_cast %[[INDEX_1]] : index to i32
// CHECK:         arith.index_cast %[[INDEX_2]] : index to i32
func.func @pack_multiple_linalg_indices(%input: tensor<8x5x16xi32>,
                                        %init: tensor<8x5x16xi32>) -> tensor<8x5x16xi32> {
  %result = linalg.generic {
      indexing_maps = [#identity_3d, #identity_3d],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%input : tensor<8x5x16xi32>) outs(%init : tensor<8x5x16xi32>) {
  ^bb0(%in: i32, %out: i32):
    %index_0 = linalg.index 0 : index
    %index_1 = linalg.index 1 : index
    %index_2 = linalg.index 2 : index
    %index_0_i32 = arith.index_cast %index_0 : index to i32
    %index_1_i32 = arith.index_cast %index_1 : index to i32
    %index_2_i32 = arith.index_cast %index_2 : index to i32
    %sum_0 = arith.addi %in, %index_0_i32 : i32
    %sum_1 = arith.addi %sum_0, %index_1_i32 : i32
    %sum_2 = arith.addi %sum_1, %index_2_i32 : i32
    linalg.yield %sum_2 : i32
  } -> tensor<8x5x16xi32>
  return %result : tensor<8x5x16xi32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %generic = transform.structured.match ops{["linalg.generic"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    transform.structured.pack %generic packed_sizes = [4, 0, 8]
      : (!transform.any_op) -> (!transform.op<"linalg.generic">)
    transform.yield
  }
}

// -----

#identity_dynamic = affine_map<(d0) -> (d0)>

// CHECK-DAG: #[[$DYNAMIC_PACK_INDEX:.*]] = affine_map<()[s0, s1, s2] -> (s0 * s1 + s2)>
// CHECK-LABEL: func.func @pack_linalg_index_dynamic
// CHECK:         %[[TILE_SIZE:.*]] = "tile_size"() : () -> index
// CHECK:         linalg.generic
// CHECK:       ^bb0
// CHECK:         %[[OUTER:.*]] = linalg.index 0 : index
// CHECK:         %[[INNER:.*]] = linalg.index 1 : index
// CHECK:         %[[INDEX:.*]] = affine.apply #[[$DYNAMIC_PACK_INDEX]]()[%[[TILE_SIZE]], %[[OUTER]], %[[INNER]]]
// CHECK:         arith.index_cast %[[INDEX]] : index to i32
func.func @pack_linalg_index_dynamic(%input: tensor<?xi32>,
                                     %init: tensor<?xi32>) -> tensor<?xi32> {
  %tile_size = "tile_size"() : () -> index
  %result = linalg.generic {
      indexing_maps = [#identity_dynamic, #identity_dynamic],
      iterator_types = ["parallel"]}
      ins(%input : tensor<?xi32>) outs(%init : tensor<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %index = linalg.index 0 : index
    %index_i32 = arith.index_cast %index : index to i32
    %sum = arith.addi %in, %index_i32 : i32
    linalg.yield %sum : i32
  } -> tensor<?xi32>
  return %result : tensor<?xi32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %generic = transform.structured.match ops{["linalg.generic"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %tile_size = transform.structured.match ops{["tile_size"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    transform.structured.pack %generic packed_sizes = [%tile_size]
      : (!transform.any_op, !transform.any_op) -> (!transform.op<"linalg.generic">)
    transform.yield
  }
}

// -----

//===----------------------------------------------------------------------===//
// operands_to_pack
//
// Only the listed operands are packed. The others keep their layout: each
// packed dim of size D is reshaped with a tensor.expand_shape into D/t x t
// (t being the packed size) and an init is collapsed back after the op.
//===----------------------------------------------------------------------===//

// C (operand 2) is not packed.

//    CHECK-DAG: #[[$MAP_A:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
//    CHECK-DAG: #[[$MAP_B:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d1, d4, d5)>
//    CHECK-DAG: #[[$MAP_C:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d1, d4)>

// CHECK-LABEL: matmul_skip_acc
//  CHECK-SAME:   %[[A:.+]]: tensor<24x16xf32>, %[[B:.+]]: tensor<16x16xf32>, %[[C:.+]]: tensor<24x16xf32>
func.func @matmul_skip_acc(%A: tensor<24x16xf32>, %B: tensor<16x16xf32>,
                           %C: tensor<24x16xf32>) -> tensor<24x16xf32> {
  //      CHECK: %[[PACK_A:.*]] = linalg.pack %[[A]] inner_dims_pos = [0, 1] inner_tiles = [12, 1]
  // CHECK-SAME:   : tensor<24x16xf32> -> tensor<2x16x12x1xf32>
  //      CHECK: %[[PACK_B:.*]] = linalg.pack %[[B]] inner_dims_pos = [1, 0] inner_tiles = [8, 1]
  // CHECK-SAME:   : tensor<16x16xf32> -> tensor<16x2x8x1xf32>
  //      CHECK: %[[EXPANDED_C:.*]] = tensor.expand_shape %[[C]] {{\[\[0, 1\], \[2, 3\]\]}}
  // CHECK-SAME:   output_shape [2, 12, 2, 8] : tensor<24x16xf32> into tensor<2x12x2x8xf32>
  //      CHECK: %[[RES:.*]] = linalg.generic
  // CHECK-SAME:   indexing_maps = [#[[$MAP_A]], #[[$MAP_B]], #[[$MAP_C]]]
  // CHECK-SAME:   iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]
  // CHECK-SAME:   ins(%[[PACK_A]], %[[PACK_B]] : tensor<2x16x12x1xf32>, tensor<16x2x8x1xf32>)
  // CHECK-SAME:   outs(%[[EXPANDED_C]] : tensor<2x12x2x8xf32>)
  //      CHECK: %[[COLLAPSED_C:.*]] = tensor.collapse_shape %[[RES]] {{\[\[0, 1\], \[2, 3\]\]}}
  // CHECK-SAME:   : tensor<2x12x2x8xf32> into tensor<24x16xf32>
  //      CHECK: return %[[COLLAPSED_C]]
  %0 = linalg.matmul ins(%A, %B : tensor<24x16xf32>, tensor<16x16xf32>)
                     outs(%C : tensor<24x16xf32>) -> tensor<24x16xf32>
  return %0 : tensor<24x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.pack %0 packed_sizes = [12, 8, 1] operands_to_pack = [0, 1]
        : (!transform.any_op) -> (!transform.op<"linalg.generic">)
    transform.yield
  }
}

// -----

// A (operand 0) is not packed. Being an input, nothing is collapsed after the
// op.

//    CHECK-DAG: #[[$MAP_A:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d2, d5)>
//    CHECK-DAG: #[[$MAP_B:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d1, d4, d5)>
//    CHECK-DAG: #[[$MAP_C:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>

// CHECK-LABEL: matmul_skip_lhs
//  CHECK-SAME:   %[[A:.+]]: tensor<24x16xf32>, %[[B:.+]]: tensor<16x16xf32>, %[[C:.+]]: tensor<24x16xf32>
func.func @matmul_skip_lhs(%A: tensor<24x16xf32>, %B: tensor<16x16xf32>,
                           %C: tensor<24x16xf32>) -> tensor<24x16xf32> {
  //      CHECK: %[[EXPANDED_A:.*]] = tensor.expand_shape %[[A]] {{\[\[0, 1\], \[2, 3\]\]}}
  // CHECK-SAME:   output_shape [2, 12, 4, 4] : tensor<24x16xf32> into tensor<2x12x4x4xf32>
  //      CHECK: %[[PACK_B:.*]] = linalg.pack %[[B]] inner_dims_pos = [1, 0] inner_tiles = [8, 4]
  // CHECK-SAME:   : tensor<16x16xf32> -> tensor<4x2x8x4xf32>
  //      CHECK: %[[PACK_C:.*]] = linalg.pack %[[C]] inner_dims_pos = [0, 1] inner_tiles = [12, 8]
  // CHECK-SAME:   : tensor<24x16xf32> -> tensor<2x2x12x8xf32>
  //      CHECK: %[[RES:.*]] = linalg.generic
  // CHECK-SAME:   indexing_maps = [#[[$MAP_A]], #[[$MAP_B]], #[[$MAP_C]]]
  // CHECK-SAME:   ins(%[[EXPANDED_A]], %[[PACK_B]] : tensor<2x12x4x4xf32>, tensor<4x2x8x4xf32>)
  // CHECK-SAME:   outs(%[[PACK_C]] : tensor<2x2x12x8xf32>)
  //  CHECK-NOT: tensor.collapse_shape
  //      CHECK: %[[UNPACK_C:.*]] = linalg.unpack %[[RES]] inner_dims_pos = [0, 1] inner_tiles = [12, 8] into %[[C]]
  // CHECK-SAME:   : tensor<2x2x12x8xf32> -> tensor<24x16xf32>
  //      CHECK: return %[[UNPACK_C]]
  %0 = linalg.matmul ins(%A, %B : tensor<24x16xf32>, tensor<16x16xf32>)
                     outs(%C : tensor<24x16xf32>) -> tensor<24x16xf32>
  return %0 : tensor<24x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.pack %0 packed_sizes = [12, 8, 4] operands_to_pack = [1, 2]
        : (!transform.any_op) -> (!transform.op<"linalg.generic">)
    transform.yield
  }
}

// -----

// B (operand 1) is not packed and is indexed by the transposed map (N, K).
// Its N and K dims are expanded in place, so its inner dims follow the order
// of B (n before k).

//    CHECK-DAG: #[[$MAP_A:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
//    CHECK-DAG: #[[$MAP_B:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d4, d2, d5)>
//    CHECK-DAG: #[[$MAP_C:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>

// CHECK-LABEL: matmul_transpose_b_skip_rhs
//  CHECK-SAME:   %[[A:.+]]: tensor<24x32xf32>, %[[B:.+]]: tensor<16x32xf32>, %[[C:.+]]: tensor<24x16xf32>
func.func @matmul_transpose_b_skip_rhs(%A: tensor<24x32xf32>, %B: tensor<16x32xf32>,
                                       %C: tensor<24x16xf32>) -> tensor<24x16xf32> {
  //      CHECK: %[[PACK_A:.*]] = linalg.pack %[[A]] inner_dims_pos = [0, 1] inner_tiles = [12, 4]
  // CHECK-SAME:   : tensor<24x32xf32> -> tensor<2x8x12x4xf32>
  //      CHECK: %[[EXPANDED_B:.*]] = tensor.expand_shape %[[B]] {{\[\[0, 1\], \[2, 3\]\]}}
  // CHECK-SAME:   output_shape [2, 8, 8, 4] : tensor<16x32xf32> into tensor<2x8x8x4xf32>
  //      CHECK: %[[PACK_C:.*]] = linalg.pack %[[C]] inner_dims_pos = [0, 1] inner_tiles = [12, 8]
  // CHECK-SAME:   : tensor<24x16xf32> -> tensor<2x2x12x8xf32>
  //      CHECK: %[[RES:.*]] = linalg.generic
  // CHECK-SAME:   indexing_maps = [#[[$MAP_A]], #[[$MAP_B]], #[[$MAP_C]]]
  // CHECK-SAME:   ins(%[[PACK_A]], %[[EXPANDED_B]] : tensor<2x8x12x4xf32>, tensor<2x8x8x4xf32>)
  // CHECK-SAME:   outs(%[[PACK_C]] : tensor<2x2x12x8xf32>)
  //      CHECK: linalg.unpack %[[RES]] inner_dims_pos = [0, 1] inner_tiles = [12, 8] into %[[C]]
  %0 = linalg.matmul
      indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                       affine_map<(m, n, k) -> (n, k)>,
                       affine_map<(m, n, k) -> (m, n)>]
      ins(%A, %B : tensor<24x32xf32>, tensor<16x32xf32>)
      outs(%C : tensor<24x16xf32>) -> tensor<24x16xf32>
  return %0 : tensor<24x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.pack %0 packed_sizes = [12, 8, 4] operands_to_pack = [0, 2]
        : (!transform.any_op) -> (!transform.op<"linalg.generic">)
    transform.yield
  }
}

// -----

// An empty operands_to_pack packs no operand: all of them are expanded.

//    CHECK-DAG: #[[$MAP_A:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>
//    CHECK-DAG: #[[$MAP_B:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d2, d4, d1, d3)>
//    CHECK-DAG: #[[$MAP_C:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>

// CHECK-LABEL: matmul_pack_none
//  CHECK-SAME:   %[[A:.+]]: tensor<24x16xf32>, %[[B:.+]]: tensor<16x16xf32>, %[[C:.+]]: tensor<24x16xf32>
func.func @matmul_pack_none(%A: tensor<24x16xf32>, %B: tensor<16x16xf32>,
                            %C: tensor<24x16xf32>) -> tensor<24x16xf32> {
  //  CHECK-NOT: linalg.pack
  //      CHECK: %[[EXPANDED_A:.*]] = tensor.expand_shape %[[A]] {{\[\[0\], \[1, 2\]\]}}
  // CHECK-SAME:   output_shape [24, 4, 4] : tensor<24x16xf32> into tensor<24x4x4xf32>
  //      CHECK: %[[EXPANDED_B:.*]] = tensor.expand_shape %[[B]] {{\[\[0, 1\], \[2, 3\]\]}}
  // CHECK-SAME:   output_shape [4, 4, 2, 8] : tensor<16x16xf32> into tensor<4x4x2x8xf32>
  //      CHECK: %[[EXPANDED_C:.*]] = tensor.expand_shape %[[C]] {{\[\[0\], \[1, 2\]\]}}
  // CHECK-SAME:   output_shape [24, 2, 8] : tensor<24x16xf32> into tensor<24x2x8xf32>
  //      CHECK: %[[RES:.*]] = linalg.generic
  // CHECK-SAME:   indexing_maps = [#[[$MAP_A]], #[[$MAP_B]], #[[$MAP_C]]]
  // CHECK-SAME:   iterator_types = ["parallel", "parallel", "reduction", "parallel", "reduction"]
  // CHECK-SAME:   ins(%[[EXPANDED_A]], %[[EXPANDED_B]] : tensor<24x4x4xf32>, tensor<4x4x2x8xf32>)
  // CHECK-SAME:   outs(%[[EXPANDED_C]] : tensor<24x2x8xf32>)
  //      CHECK: %[[COLLAPSED_C:.*]] = tensor.collapse_shape %[[RES]] {{\[\[0\], \[1, 2\]\]}}
  // CHECK-SAME:   : tensor<24x2x8xf32> into tensor<24x16xf32>
  //  CHECK-NOT: linalg.unpack
  //      CHECK: return %[[COLLAPSED_C]]
  %0 = linalg.matmul ins(%A, %B : tensor<24x16xf32>, tensor<16x16xf32>)
                     outs(%C : tensor<24x16xf32>) -> tensor<24x16xf32>
  return %0 : tensor<24x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.pack %0 packed_sizes = [0, 8, 4] operands_to_pack = []
        : (!transform.any_op) -> (!transform.op<"linalg.generic">)
    transform.yield
  }
}

// -----

// M (25) is not a multiple of its packed size (12), so C, which is not
// packed, cannot be expanded (only linalg.pack can pad).
// assume_dynamic_dims_divisible has no effect on static dims.

func.func @matmul_skip_acc_m_not_a_multiple(%A: tensor<25x16xf32>, %B: tensor<16x16xf32>,
                                            %C: tensor<25x16xf32>) -> tensor<25x16xf32> {
  %0 = linalg.matmul ins(%A, %B : tensor<25x16xf32>, tensor<16x16xf32>)
                     outs(%C : tensor<25x16xf32>) -> tensor<25x16xf32>
  return %0 : tensor<25x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{data tiling failed}}
    %1 = transform.structured.pack %0 packed_sizes = [12, 8, 1] operands_to_pack = [0, 1]
        assume_dynamic_dims_divisible
        : (!transform.any_op) -> (!transform.op<"linalg.generic">)
    transform.yield
  }
}

// -----

// M is dynamic, so whether it is a multiple of its packed size (12) cannot be
// checked. Without assume_dynamic_dims_divisible, C cannot be expanded.

func.func @matmul_skip_acc_dynamic_m(%A: tensor<?x16xf32>, %B: tensor<16x16xf32>,
                                     %C: tensor<?x16xf32>) -> tensor<?x16xf32> {
  %0 = linalg.matmul ins(%A, %B : tensor<?x16xf32>, tensor<16x16xf32>)
                     outs(%C : tensor<?x16xf32>) -> tensor<?x16xf32>
  return %0 : tensor<?x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{data tiling failed}}
    %1 = transform.structured.pack %0 packed_sizes = [12, 8, 0] operands_to_pack = [0, 1]
        : (!transform.any_op) -> (!transform.op<"linalg.generic">)
    transform.yield
  }
}

// -----

// M is dynamic and assume_dynamic_dims_divisible asserts that it is a multiple
// of its packed size (12), so C is expanded.

// CHECK-LABEL: matmul_skip_acc_dynamic_m_assume_divisible
//  CHECK-SAME:   %[[A:.+]]: tensor<?x16xf32>, %[[B:.+]]: tensor<16x16xf32>, %[[C:.+]]: tensor<?x16xf32>
func.func @matmul_skip_acc_dynamic_m_assume_divisible(%A: tensor<?x16xf32>, %B: tensor<16x16xf32>,
                                                      %C: tensor<?x16xf32>) -> tensor<?x16xf32> {
  //      CHECK: %[[PACK_A:.*]] = linalg.pack %[[A]] padding_value(%{{.*}} : f32) inner_dims_pos = [0] inner_tiles = [12]
  // CHECK-SAME:   : tensor<?x16xf32> -> tensor<?x16x12xf32>
  //      CHECK: %[[PACK_B:.*]] = linalg.pack %[[B]] inner_dims_pos = [1] inner_tiles = [8]
  // CHECK-SAME:   : tensor<16x16xf32> -> tensor<16x2x8xf32>
  //      CHECK: %[[M:.*]] = tensor.dim %[[C]], %{{.*}} : tensor<?x16xf32>
  //      CHECK: %[[C12:.*]] = arith.constant 12 : index
  //      CHECK: %[[OUTER_M:.*]] = arith.divsi %[[M]], %[[C12]] : index
  //      CHECK: %[[EXPANDED_C:.*]] = tensor.expand_shape %[[C]] {{\[\[0, 1\], \[2, 3\]\]}}
  // CHECK-SAME:   output_shape [%[[OUTER_M]], 12, 2, 8] : tensor<?x16xf32> into tensor<?x12x2x8xf32>
  //      CHECK: %[[RES:.*]] = linalg.generic
  // CHECK-SAME:   ins(%[[PACK_A]], %[[PACK_B]] : tensor<?x16x12xf32>, tensor<16x2x8xf32>)
  // CHECK-SAME:   outs(%[[EXPANDED_C]] : tensor<?x12x2x8xf32>)
  //      CHECK: %[[COLLAPSED_C:.*]] = tensor.collapse_shape %[[RES]] {{\[\[0, 1\], \[2, 3\]\]}}
  // CHECK-SAME:   : tensor<?x12x2x8xf32> into tensor<?x16xf32>
  //      CHECK: return %[[COLLAPSED_C]]
  %0 = linalg.matmul ins(%A, %B : tensor<?x16xf32>, tensor<16x16xf32>)
                     outs(%C : tensor<?x16xf32>) -> tensor<?x16xf32>
  return %0 : tensor<?x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.pack %0 packed_sizes = [12, 8, 0] operands_to_pack = [0, 1]
        assume_dynamic_dims_divisible
        : (!transform.any_op) -> (!transform.op<"linalg.generic">)
    transform.yield
  }
}

// -----

// M is dynamic but its packed size is 1, so it is always a multiple and C is
// expanded without assume_dynamic_dims_divisible.

// CHECK-LABEL: matmul_skip_acc_dynamic_m_unit_tile
//  CHECK-SAME:   %[[A:.+]]: tensor<?x16xf32>, %[[B:.+]]: tensor<16x16xf32>, %[[C:.+]]: tensor<?x16xf32>
func.func @matmul_skip_acc_dynamic_m_unit_tile(%A: tensor<?x16xf32>, %B: tensor<16x16xf32>,
                                               %C: tensor<?x16xf32>) -> tensor<?x16xf32> {
  //      CHECK: %[[PACK_A:.*]] = linalg.pack %[[A]] padding_value(%{{.*}} : f32) inner_dims_pos = [0] inner_tiles = [1]
  // CHECK-SAME:   : tensor<?x16xf32> -> tensor<?x16x1xf32>
  //      CHECK: %[[PACK_B:.*]] = linalg.pack %[[B]] inner_dims_pos = [1] inner_tiles = [8]
  // CHECK-SAME:   : tensor<16x16xf32> -> tensor<16x2x8xf32>
  //      CHECK: %[[M:.*]] = tensor.dim %[[C]], %{{.*}} : tensor<?x16xf32>
  //      CHECK: %[[EXPANDED_C:.*]] = tensor.expand_shape %[[C]] {{\[\[0, 1\], \[2, 3\]\]}}
  // CHECK-SAME:   output_shape [%[[M]], 1, 2, 8] : tensor<?x16xf32> into tensor<?x1x2x8xf32>
  //      CHECK: %[[RES:.*]] = linalg.generic
  // CHECK-SAME:   ins(%[[PACK_A]], %[[PACK_B]] : tensor<?x16x1xf32>, tensor<16x2x8xf32>)
  // CHECK-SAME:   outs(%[[EXPANDED_C]] : tensor<?x1x2x8xf32>)
  //      CHECK: %[[COLLAPSED_C:.*]] = tensor.collapse_shape %[[RES]] {{\[\[0, 1\], \[2, 3\]\]}}
  // CHECK-SAME:   : tensor<?x1x2x8xf32> into tensor<?x16xf32>
  //      CHECK: return %[[COLLAPSED_C]]
  %0 = linalg.matmul ins(%A, %B : tensor<?x16xf32>, tensor<16x16xf32>)
                     outs(%C : tensor<?x16xf32>) -> tensor<?x16xf32>
  return %0 : tensor<?x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.pack %0 packed_sizes = [1, 8, 0] operands_to_pack = [0, 1]
        : (!transform.any_op) -> (!transform.op<"linalg.generic">)
    transform.yield
  }
}

// -----

// The packed size of K is only known at runtime. Only A and B use K and both
// are packed, so C, which is not packed, is expanded by the static packed size
// of N only.

// CHECK-LABEL: matmul_skip_acc_dynamic_pack_size
//  CHECK-SAME:   %[[A:.+]]: tensor<24x16xf32>, %[[B:.+]]: tensor<16x16xf32>, %[[C:.+]]: tensor<24x16xf32>
func.func @matmul_skip_acc_dynamic_pack_size(%A: tensor<24x16xf32>, %B: tensor<16x16xf32>,
                                             %C: tensor<24x16xf32>) -> tensor<24x16xf32> {
  //      CHECK: %[[TS:.*]] = "some_tile_size"() : () -> index
  //      CHECK: %[[PACK_A:.*]] = linalg.pack %[[A]] padding_value(%{{.*}} : f32) inner_dims_pos = [1] inner_tiles = [%[[TS]]]
  // CHECK-SAME:   : tensor<24x16xf32> -> tensor<24x?x?xf32>
  //      CHECK: %[[PACK_B:.*]] = linalg.pack %[[B]] padding_value(%{{.*}} : f32) inner_dims_pos = [1, 0] inner_tiles = [8, %[[TS]]]
  // CHECK-SAME:   : tensor<16x16xf32> -> tensor<?x2x8x?xf32>
  //      CHECK: %[[EXPANDED_C:.*]] = tensor.expand_shape %[[C]] {{\[\[0\], \[1, 2\]\]}}
  // CHECK-SAME:   output_shape [24, 2, 8] : tensor<24x16xf32> into tensor<24x2x8xf32>
  //      CHECK: %[[RES:.*]] = linalg.generic
  // CHECK-SAME:   ins(%[[PACK_A]], %[[PACK_B]] : tensor<24x?x?xf32>, tensor<?x2x8x?xf32>)
  // CHECK-SAME:   outs(%[[EXPANDED_C]] : tensor<24x2x8xf32>)
  //      CHECK: %[[COLLAPSED_C:.*]] = tensor.collapse_shape %[[RES]] {{\[\[0\], \[1, 2\]\]}}
  // CHECK-SAME:   : tensor<24x2x8xf32> into tensor<24x16xf32>
  //      CHECK: return %[[COLLAPSED_C]]
  %sz = "some_tile_size"() : () -> (index)
  %0 = linalg.matmul ins(%A, %B : tensor<24x16xf32>, tensor<16x16xf32>)
                     outs(%C : tensor<24x16xf32>) -> tensor<24x16xf32>
  return %0 : tensor<24x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %sz = transform.structured.match ops{["some_tile_size"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.pack %0 packed_sizes = [0, 8, %sz] operands_to_pack = [0, 1]
        : (!transform.any_op, !transform.any_op) -> (!transform.op<"linalg.generic">)
    transform.yield
  }
}

// -----

// The packed size of K is only known at runtime and B, which uses K, is not
// packed. B cannot be expanded, even with assume_dynamic_dims_divisible, since
// its inner dim would not have a static size.

func.func @matmul_skip_rhs_dynamic_pack_size(%A: tensor<24x16xf32>, %B: tensor<16x16xf32>,
                                             %C: tensor<24x16xf32>) -> tensor<24x16xf32> {
  %sz = "some_tile_size"() : () -> (index)
  %0 = linalg.matmul ins(%A, %B : tensor<24x16xf32>, tensor<16x16xf32>)
                     outs(%C : tensor<24x16xf32>) -> tensor<24x16xf32>
  return %0 : tensor<24x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %sz = transform.structured.match ops{["some_tile_size"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{data tiling failed}}
    %1 = transform.structured.pack %0 packed_sizes = [0, 8, %sz] operands_to_pack = [0, 2]
        assume_dynamic_dims_divisible
        : (!transform.any_op, !transform.any_op) -> (!transform.op<"linalg.generic">)
    transform.yield
  }
}

// -----

// A (operand 0) is not packed but has an encoding, which is tied to its shape
// and cannot be carried over to the expanded one.

#CSR = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>

func.func @matmul_skip_lhs_encoding(%A: tensor<24x16xf32, #CSR>, %B: tensor<16x16xf32>,
                                    %C: tensor<24x16xf32>) -> tensor<24x16xf32> {
  %0 = linalg.matmul ins(%A, %B : tensor<24x16xf32, #CSR>, tensor<16x16xf32>)
                     outs(%C : tensor<24x16xf32>) -> tensor<24x16xf32>
  return %0 : tensor<24x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{data tiling failed}}
    %1 = transform.structured.pack %0 packed_sizes = [2, 0, 0] operands_to_pack = [1, 2]
        : (!transform.any_op) -> (!transform.op<"linalg.generic">)
    transform.yield
  }
}

// -----

// Operand number 3 is out of range (the matmul has 3 operands).

func.func @matmul_operands_to_pack_out_of_range(%A: tensor<24x16xf32>, %B: tensor<16x16xf32>,
                                                %C: tensor<24x16xf32>) -> tensor<24x16xf32> {
  %0 = linalg.matmul ins(%A, %B : tensor<24x16xf32>, tensor<16x16xf32>)
                     outs(%C : tensor<24x16xf32>) -> tensor<24x16xf32>
  return %0 : tensor<24x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{data tiling failed}}
    %1 = transform.structured.pack %0 packed_sizes = [12, 8, 1] operands_to_pack = [3]
        : (!transform.any_op) -> (!transform.op<"linalg.generic">)
    transform.yield
  }
}
