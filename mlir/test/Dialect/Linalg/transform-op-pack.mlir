// RUN: mlir-opt -test-transform-dialect-interpreter -split-input-file -verify-diagnostics -allow-unregistered-dialect %s | FileCheck %s

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
  //      CHECK: %[[PACKED:.*]] = tensor.pack %[[T0]] padding_value(%{{.*}} : f16) 
  // CHECK-SAME:   inner_dims_pos = [1] inner_tiles = [4] into %[[EMPTY]] : tensor<3x7xf16> -> tensor<3x2x4xf16>
  //  CHECK-NOT: tensor.pack
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

  //  CHECK-NOT: tensor.unpack
  return %2 : tensor<3xf16>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.generic"]} in %arg1
  transform.structured.pack %0 packed_sizes = [0, 4]
      : (!pdl.operation) -> (!transform.op<"linalg.generic">)
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#col_reduction_2d_trait = {
  indexing_maps = [#map, #map1],
  iterator_types = ["reduction", "parallel"]
}

//    CHECK-DAG: #[[$PACKED_MAP_0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//    CHECK-DAG: #[[$PACKED_MAP_1:.*]] = affine_map<(d0, d1, d2) -> (d1)>

//  CHECK-LABEL: @col_reduction_2d_static
//   CHECK-SAME:   %[[T0:.+]]: tensor<7x3xf16>,
//   CHECK-SAME:   %[[T1:.+]]: tensor<3xf16>
func.func @col_reduction_2d_static(%t0: tensor<7x3xf16>, %t1: tensor<3xf16>) -> tensor<3xf16> {
  //      CHECK:  %[[EMPTY:.*]] = tensor.empty() : tensor<2x3x4xf16>
  //      CHECK: %[[PACKED:.*]] = tensor.pack %[[T0]] padding_value(%{{.*}} : f16) 
  // CHECK-SAME:   inner_dims_pos = [0] inner_tiles = [4] into %[[EMPTY]] : tensor<7x3xf16> -> tensor<2x3x4xf16>
  //  CHECK-NOT: tensor.pack
  //      CHECK: linalg.generic 
  // CHECK-SAME:   indexing_maps = [#[[$PACKED_MAP_0]], #[[$PACKED_MAP_1]]]
  // CHECK-SAME:   iterator_types = ["reduction", "parallel", "reduction"]
  // CHECK-SAME:   ins(%{{.*}} : tensor<2x3x4xf16>)
  // CHECK-SAME:  outs(%{{.*}} : tensor<3xf16>)
  %2 = linalg.generic #col_reduction_2d_trait ins(%t0 : tensor<7x3xf16>) outs(%t1 : tensor<3xf16>) {
  ^bb0(%in: f16, %out: f16):
    %3 = arith.addf %in, %out : f16
    linalg.yield %3 : f16
  } -> tensor<3xf16>

  //  CHECK-NOT: tensor.unpack
  return %2 : tensor<3xf16>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.generic"]} in %arg1
  transform.structured.pack %0 packed_sizes = [4, 0]
      : (!pdl.operation) -> (!transform.op<"linalg.generic">)
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
  //      CHECK: %[[PACKED:.*]] = tensor.pack %[[T0]] padding_value(%{{.*}} : f16) 
  // CHECK-SAME:   inner_dims_pos = [1] inner_tiles = [4] into %[[EMPTY]] : tensor<?x?xf16> -> tensor<?x?x4xf16>
  //  CHECK-NOT: tensor.pack
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

  //  CHECK-NOT: tensor.unpack
  return %2 : tensor<?xf16>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.generic"]} in %arg1
  transform.structured.pack %0 packed_sizes = [0, 4]
      : (!pdl.operation) -> (!transform.op<"linalg.generic">)
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
  //      CHECK: %[[PACKED_0:.*]] = tensor.pack %[[T0]] padding_value(%{{.*}} : f16) 
  // CHECK-SAME:   inner_dims_pos = [0, 1] inner_tiles = [3, 4] into %{{.*}} : tensor<?x?xf16> -> tensor<?x?x3x4xf16>
  //      CHECK: %[[PACKED_1:.*]] = tensor.pack %[[T1]] padding_value(%{{.*}} : f16) 
  // CHECK-SAME:   inner_dims_pos = [0] inner_tiles = [3] into %{{.*}} : tensor<?xf16> -> tensor<?x3xf16>
  //  CHECK-NOT: tensor.pack
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

  //      CHECK: tensor.unpack %{{.*}} inner_dims_pos = [0] inner_tiles = [3] into %{{.*}} : tensor<?x3xf16> -> tensor<?xf16>
  return %2 : tensor<?xf16>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.generic"]} in %arg1
  transform.structured.pack %0 packed_sizes = [3, 4]
      : (!pdl.operation) -> (!transform.op<"linalg.generic">)
}

// -----

//                                                M   N   K   m   n   k       M   K   m   k
// CHECK-DAG: #[[$PACKED_MAP_0:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
//                                                                            K   N   n   k
// CHECK-DAG: #[[$PACKED_MAP_1:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d1, d4, d5)>
//                                                                            M   N   m   n
// CHECK-DAG: #[[$PACKED_MAP_2:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>

// CHECK-LABEL: @matmul
//  CHECK-SAME:   %[[A:[0-9a-zA-Z]+]]: tensor<?x?xf32>,
//  CHECK-SAME:   %[[B:[0-9a-zA-Z]+]]: tensor<?x?xf32>,
//  CHECK-SAME:   %[[C:[0-9a-zA-Z]+]]: tensor<?x?xf32>
func.func @matmul(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>)
    -> tensor<?x?xf32> {

  //      CHECK: %[[PACK_A:.*]] = tensor.pack %{{.*}} inner_dims_pos = [0, 1] inner_tiles = [2, 4]
  // CHECK-SAME:   : tensor<?x?xf32> -> tensor<?x?x2x4xf32>
  //      CHECK: %[[PACK_B:.*]] = tensor.pack %{{.*}} inner_dims_pos = [1, 0] inner_tiles = [3, 4]
  // CHECK-SAME:   : tensor<?x?xf32> -> tensor<?x?x3x4xf32>
  //      CHECK: %[[PACK_C:.*]] = tensor.pack %{{.*}} inner_dims_pos = [0, 1] inner_tiles = [2, 3]
  // CHECK-SAME:   : tensor<?x?xf32> -> tensor<?x?x2x3xf32>

  //      CHECK: linalg.generic {indexing_maps = [#[[$PACKED_MAP_0]], #[[$PACKED_MAP_1]], #[[$PACKED_MAP_2]]]
  // CHECK-SAME:     iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} 
  // CHECK-SAME:  ins(%{{.*}} : tensor<?x?x2x4xf32>, tensor<?x?x3x4xf32>)
  // CHECK-SAME: outs(%{{.*}} : tensor<?x?x2x3xf32>)
  %0 = linalg.matmul  ins(%A, %B: tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%C: tensor<?x?xf32>)
    -> tensor<?x?xf32>

  //      CHECK: tensor.unpack %{{.*}} inner_dims_pos = [0, 1] inner_tiles = [2, 3]
  // CHECK-SAME:   : tensor<?x?x2x3xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
    //                                            M  N  K
    %1 = transform.structured.pack %0 packed_sizes = [2, 3, 4]
      : (!pdl.operation) -> (!transform.op<"linalg.generic">)
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

  //      CHECK: %[[PACK_INPUT:.*]] = tensor.pack %{{.*}} inner_dims_pos = [1] inner_tiles = [8]
  // CHECK-SAME:   : tensor<14x512x28x28xf32> -> tensor<14x64x28x28x8xf32>
  //      CHECK: %[[PACK_FILTER:.*]] = tensor.pack %{{.*}} inner_dims_pos = [0, 1] inner_tiles = [4, 8]
  // CHECK-SAME:   : tensor<1024x512x1x1xf32> -> tensor<256x64x1x1x4x8xf32>
  //      CHECK: %[[PACK_INPUT:.*]] = tensor.pack %{{.*}} inner_dims_pos = [1] inner_tiles = [4]
  // CHECK-SAME:   : tensor<14x1024x28x28xf32> -> tensor<14x256x28x28x4xf32>
  //      CHECK: linalg.generic {indexing_maps = [#[[$PACKED_MAP_0]], #[[$PACKED_MAP_1]], #[[$PACKED_MAP_2]]]
  // CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "parallel", "reduction"]} 
  // CHECK-SAME:  ins(%{{.*}} : tensor<14x64x28x28x8xf32>, tensor<256x64x1x1x4x8xf32>)
  // CHECK-SAME: outs(%{{.*}} : tensor<14x256x28x28x4xf32>)
  %0 = linalg.conv_2d_nchw_fchw ins(%i, %f: tensor<14x512x28x28xf32>, tensor<1024x512x1x1xf32>)
                                outs(%o: tensor<14x1024x28x28xf32>) -> tensor<14x1024x28x28xf32>

  //      CHECK: tensor.unpack %{{.*}} inner_dims_pos = [1] inner_tiles = [4]
  // CHECK-SAME:   : tensor<14x256x28x28x4xf32> -> tensor<14x1024x28x28xf32>
  return %0: tensor<14x1024x28x28xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match interface{LinalgOp} in %arg1
  //                                            N  F  H  W  C KH KW
  %1 = transform.structured.pack %0 packed_sizes = [0, 4, 0, 0, 8, 0, 0]
      : (!pdl.operation) -> (!transform.op<"linalg.generic">)
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
  
  //      CHECK: %[[PACK_INPUT:.*]] = tensor.pack %{{.*}} inner_dims_pos = [3] inner_tiles = [6]
  // CHECK-SAME:   : tensor<?x1x?x?xf32> -> tensor<?x1x?x?x6xf32>
  //      CHECK: %[[PACK_FILTER:.*]] = tensor.pack %{{.*}} inner_dims_pos = [3, 2] inner_tiles = [4, 6]
  // CHECK-SAME:   : tensor<1x?x?x?xf32> -> tensor<1x?x?x?x4x6xf32>
  //      CHECK: %[[PACK_OUTPUT:.*]] = tensor.pack %{{.*}} inner_dims_pos = [3] inner_tiles = [4]
  // CHECK-SAME:   : tensor<?x1x?x?xf32> -> tensor<?x1x?x?x4xf32>

  //      CHECK: linalg.generic {indexing_maps = [#[[$PACKED_MAP_0]], #[[$PACKED_MAP_1]], #[[$PACKED_MAP_2]]]
  // CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "parallel", "reduction"]} 
  // CHECK-SAME:  ins(%{{.*}} : tensor<?x1x?x?x6xf32>, tensor<1x?x?x?x4x6xf32>)
  // CHECK-SAME: outs(%{{.*}} : tensor<?x1x?x?x4xf32>)
  %0 = linalg.conv_2d_nhwc_hwcf
     ins (%input, %filter: tensor<?x1x?x?xf32>, tensor<1x?x?x?xf32>)
    outs (%init: tensor<?x1x?x?xf32>) -> tensor<?x1x?x?xf32>
  
  //      CHECK: tensor.unpack %{{.*}} inner_dims_pos = [3] inner_tiles = [4]
  // CHECK-SAME:   : tensor<?x1x?x?x4xf32> -> tensor<?x1x?x?xf32>
  return %0 : tensor<?x1x?x?xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match interface{LinalgOp} in %arg1
  //                                            N  H  W  F KH KW  C
  %1 = transform.structured.pack %0 packed_sizes = [0, 0, 0, 4, 0, 0, 6]
      : (!pdl.operation) -> (!transform.op<"linalg.generic">)
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

  //      CHECK: %[[PACK_A:.*]] = tensor.pack %[[A]] {{.*}} inner_dims_pos = [1] inner_tiles = [%[[TS]]]
  // CHECK-SAME:   : tensor<?x?xf32> -> tensor<?x?x?xf32>
  //      CHECK: %[[PACK_B:.*]] = tensor.pack %[[B]] {{.*}} inner_dims_pos = [1, 0] inner_tiles = [%[[TS]], %[[TS]]]
  // CHECK-SAME:   : tensor<?x?xf32> -> tensor<?x?x?x?xf32>
  //      CHECK: %[[PACK_C:.*]] = tensor.pack %[[C]] {{.*}} inner_dims_pos = [1] inner_tiles = [%[[TS]]]
  // CHECK-SAME:   : tensor<?x?xf32> -> tensor<?x?x?xf32>
  //      CHECK: linalg.generic {indexing_maps = [#[[$PACKED_MAP_0]], #[[$PACKED_MAP_1]], #[[$PACKED_MAP_2]]]
  // CHECK-SAME:     iterator_types = ["parallel", "parallel", "reduction", "parallel", "reduction"]} 
  // CHECK-SAME:  ins(%{{.*}} : tensor<?x?x?xf32>, tensor<?x?x?x?xf32>)
  // CHECK-SAME: outs(%{{.*}} : tensor<?x?x?xf32>)
  %0 = linalg.matmul  ins(%A, %B: tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%C: tensor<?x?xf32>)
    -> tensor<?x?xf32>

  //      CHECK: tensor.unpack %{{.*}} inner_dims_pos = [1] inner_tiles = [%[[TS]]] into %[[C]]
  // CHECK-SAME:   : tensor<?x?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
    %sz = transform.structured.match ops{["some_tile_size"]} in %arg1
    %1 = transform.structured.pack %0 packed_sizes = [0, %sz, %sz] 
      : (!pdl.operation) -> (!transform.op<"linalg.generic">)
}

// -----

func.func @conv_cant_pack(%i: tensor<14x512x28x28xf32>, %f: tensor<1024x512x1x1xf32>,
                          %o: tensor<14x1024x28x28xf32>) -> tensor<14x1024x28x28xf32> {
  %0 = linalg.conv_2d_nchw_fchw ins(%i, %f: tensor<14x512x28x28xf32>, tensor<1024x512x1x1xf32>)
                                outs(%o: tensor<14x1024x28x28xf32>) -> tensor<14x1024x28x28xf32>
  return %0: tensor<14x1024x28x28xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match interface{LinalgOp} in %arg1
  //                                                N  F  H  W  C KH KW
  // expected-error @below {{data tiling failed}}
  %1 = transform.structured.pack %0 packed_sizes = [0, 0, 4, 0, 0, 0, 0]
      : (!pdl.operation) -> (!transform.op<"linalg.generic">)
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

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
    // expected-error @below {{requires target to map to exactly 1 LinalgOp (got 2)}}
    %1 = transform.structured.pack %0 packed_sizes = [2, 3, 4] 
      : (!pdl.operation) -> (!transform.op<"linalg.generic">)
}


// -----

func.func @matmul(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>)
    -> tensor<?x?xf32> {
  %0 = linalg.matmul  ins(%A, %B: tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%C: tensor<?x?xf32>)
    -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
    // expected-error @below {{requires number of packed sizes match the number of loops (2 vs 3)}}
    %1 = transform.structured.pack %0 packed_sizes = [2, 3] 
      : (!pdl.operation) -> (!transform.op<"linalg.generic">)
}
