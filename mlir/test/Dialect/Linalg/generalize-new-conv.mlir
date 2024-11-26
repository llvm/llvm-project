// RUN: mlir-opt %s -split-input-file -test-linalg-new-conv -linalg-generalize-named-ops | FileCheck %s

// CHECK: #map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2 + d4)>
// CHECK: #map1 = affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>
// CHECK: #map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
// CHECK: module {
// CHECK:   func.func @conv_1d_ncw_fcw(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
// CHECK:     %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%arg2 : tensor<?x?x?xf32>) {
// CHECK:     ^bb0(%in: f32, %in_0: f32, %out: f32):
// CHECK:       %1 = arith.mulf %in, %in_0 : f32
// CHECK:       %2 = arith.addf %out, %1 : f32
// CHECK:       linalg.yield %2 : f32
// CHECK:     } -> tensor<?x?x?xf32>
// CHECK:     return %0 : tensor<?x?x?xf32>
// CHECK:   }
// CHECK: }
func.func @conv_1d_ncw_fcw(%input: tensor<?x?x?xf32>, %filter: tensor<?x?x?xf32>, %init: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = linalg.conv_1d_ncw_fcw {dilations = dense<1> : tensor<1xi64>,
                                            strides = dense<1> : tensor<1xi64>}
     ins (%input, %filter: tensor<?x?x?xf32>, tensor<?x?x?xf32>)
    outs (%init: tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}

// -----

// CHECK: #map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 + d3, d4)>
// CHECK: #map1 = affine_map<(d0, d1, d2, d3, d4) -> (d3, d4, d2)>
// CHECK: #map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
// CHECK: module {
// CHECK:   func.func @conv_1d_nwc_wcf(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
// CHECK:     %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%arg2 : tensor<?x?x?xf32>) {
// CHECK:     ^bb0(%in: f32, %in_0: f32, %out: f32):
// CHECK:       %1 = arith.mulf %in, %in_0 : f32
// CHECK:       %2 = arith.addf %out, %1 : f32
// CHECK:       linalg.yield %2 : f32
// CHECK:     } -> tensor<?x?x?xf32>
// CHECK:     return %0 : tensor<?x?x?xf32>
// CHECK:   }
// CHECK: }
func.func @conv_1d_nwc_wcf(%input: tensor<?x?x?xf32>, %filter: tensor<?x?x?xf32>, %init: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = linalg.conv_1d_nwc_wcf {dilations = dense<1> : tensor<1xi64>,
                                            strides = dense<1> : tensor<1xi64>}
     ins (%input, %filter: tensor<?x?x?xf32>, tensor<?x?x?xf32>)
    outs (%init: tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}

// -----

// CHECK: #map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d5, d3 * 2 + d6 * 3, d4 * 2 + d7 * 3)>
// CHECK: #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d2, d1, d5, d6, d7)>
// CHECK: #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>
// CHECK: module {
// CHECK:   func.func @conv_2d_ngchw_fgchw_dilated_strided(%arg0: tensor<?x?x?x?x?xf32>, %arg1: tensor<?x?x?x?x?xf32>, %arg2: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32> {
// CHECK:     %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>) outs(%arg2 : tensor<?x?x?x?x?xf32>) {
// CHECK:     ^bb0(%in: f32, %in_0: f32, %out: f32):
// CHECK:       %1 = arith.mulf %in, %in_0 : f32
// CHECK:       %2 = arith.addf %out, %1 : f32
// CHECK:       linalg.yield %2 : f32
// CHECK:     } -> tensor<?x?x?x?x?xf32>
// CHECK:     return %0 : tensor<?x?x?x?x?xf32>
// CHECK:   }
// CHECK: }
func.func @conv_2d_ngchw_fgchw_dilated_strided(%input: tensor<?x?x?x?x?xf32>, %filter: tensor<?x?x?x?x?xf32>, %init: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32> {
  %0 = linalg.conv_2d_ngchw_fgchw {dilations = dense<3> : tensor<2xi64>,
                                              strides = dense<2> : tensor<2xi64>}
     ins (%input, %filter: tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>)
    outs (%init: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
  return %0 : tensor<?x?x?x?x?xf32>
}

// -----

// CHECK: #map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 + d3, d4)>
// CHECK: #map1 = affine_map<(d0, d1, d2, d3, d4) -> (d3, d4, d2)>
// CHECK: #map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
// CHECK: module {
// CHECK:   func.func @conv_1d_nwc_wcf_memref(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>, %arg2: memref<?x?x?xf32>) {
// CHECK:     linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg0, %arg1 : memref<?x?x?xf32>, memref<?x?x?xf32>) outs(%arg2 : memref<?x?x?xf32>) {
// CHECK:     ^bb0(%in: f32, %in_0: f32, %out: f32):
// CHECK:       %0 = arith.mulf %in, %in_0 : f32
// CHECK:       %1 = arith.addf %out, %0 : f32
// CHECK:       linalg.yield %1 : f32
// CHECK:     }
// CHECK:     return
// CHECK:   }
// CHECK: }
func.func @conv_1d_nwc_wcf_memref(%input: memref<?x?x?xf32>, %filter: memref<?x?x?xf32>, %output: memref<?x?x?xf32>) {

  linalg.conv_1d_nwc_wcf {dilations = dense<1> : tensor<1xi64>,
                                       strides = dense<1> : tensor<1xi64>}
     ins (%input, %filter: memref<?x?x?xf32>, memref<?x?x?xf32>)
    outs (%output: memref<?x?x?xf32>)
  return
}

// -----

// CHECK: #map = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK: #map1 = affine_map<(d0, d1) -> (d1)>
// CHECK: #map2 = affine_map<(d0, d1) -> (d0)>
// CHECK: module {
// CHECK:   func.func @conv1d_8_tensor(%arg0: tensor<11xf32>, %arg1: tensor<4xf32>, %arg2: tensor<8xf32>) -> tensor<8xf32> {
// CHECK:     %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "reduction"]} ins(%arg0, %arg1 : tensor<11xf32>, tensor<4xf32>) outs(%arg2 : tensor<8xf32>) {
// CHECK:     ^bb0(%in: f32, %in_0: f32, %out: f32):
// CHECK:       %1 = arith.mulf %in, %in_0 : f32
// CHECK:       %2 = arith.addf %out, %1 : f32
// CHECK:       linalg.yield %2 : f32
// CHECK:     } -> tensor<8xf32>
// CHECK:     return %0 : tensor<8xf32>
// CHECK:   }
// CHECK: }
func.func @conv1d_8_tensor(%input: tensor<11xf32>, %filter: tensor<4xf32>, %output: tensor<8xf32>) -> tensor<8xf32> {
  %0 = linalg.conv_1d ins(%input, %filter : tensor<11xf32>, tensor<4xf32>)
                     outs(%output : tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

// CHECK: #map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d2 + d5, d3 + d6)>
// CHECK: #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d6)>
// CHECK: #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
// CHECK: module {
// CHECK:   func.func @batch_nchw_conv(%arg0: tensor<8x4x16x16xf32>, %arg1: tensor<16x4x3x3xf32>, %arg2: tensor<8x16x14x14xf32>) -> tensor<8x16x14x14xf32> {
// CHECK:     %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<8x4x16x16xf32>, tensor<16x4x3x3xf32>) outs(%arg2 : tensor<8x16x14x14xf32>) {
// CHECK:     ^bb0(%in: f32, %in_0: f32, %out: f32):
// CHECK:       %1 = arith.mulf %in, %in_0 : f32
// CHECK:       %2 = arith.addf %out, %1 : f32
// CHECK:       linalg.yield %2 : f32
// CHECK:     } -> tensor<8x16x14x14xf32>
// CHECK:     return %0 : tensor<8x16x14x14xf32>
// CHECK:   }
// CHECK: }
func.func @batch_nchw_conv(%arg0: tensor<8x4x16x16xf32>, %arg1: tensor<16x4x3x3xf32>, %arg2: tensor<8x16x14x14xf32>) -> tensor<8x16x14x14xf32> {
    %0 = linalg.conv_2d_nchw_fchw
      {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
       ins(%arg0, %arg1: tensor<8x4x16x16xf32>, tensor<16x4x3x3xf32>)
      outs(%arg2: tensor<8x16x14x14xf32>) -> tensor<8x16x14x14xf32>
    return %0 : tensor<8x16x14x14xf32>
}

// -----

// CHECK: #map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d5, d3 + d6, d4 + d7)>
// CHECK: #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d2, d1, d5, d6, d7)>
// CHECK: #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>
// CHECK: module {
// CHECK:   func.func @conv_2d_ngchw_fgchw(%arg0: tensor<?x?x?x?x?xf32>, %arg1: tensor<?x?x?x?x?xf32>, %arg2: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32> {
// CHECK:     %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>) outs(%arg2 : tensor<?x?x?x?x?xf32>) {
// CHECK:     ^bb0(%in: f32, %in_0: f32, %out: f32):
// CHECK:       %1 = arith.mulf %in, %in_0 : f32
// CHECK:       %2 = arith.addf %out, %1 : f32
// CHECK:       linalg.yield %2 : f32
// CHECK:     } -> tensor<?x?x?x?x?xf32>
// CHECK:     return %0 : tensor<?x?x?x?x?xf32>
// CHECK:   }
// CHECK: }
func.func @conv_2d_ngchw_fgchw(%input: tensor<?x?x?x?x?xf32>, %filter: tensor<?x?x?x?x?xf32>, %init: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32> {
  %0 = linalg.conv_2d_ngchw_fgchw {dilations = dense<1> : tensor<2xi64>,
                                              strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>)
    outs (%init: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
  return %0 : tensor<?x?x?x?x?xf32>
}

// -----

// CHECK: #map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d5, d3 + d6, d4 + d7)>
// CHECK: #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d1, d2, d5, d6, d7)>
// CHECK: #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>
// CHECK: module {
// CHECK:   func.func @conv_2d_ngchw_gfchw(%arg0: tensor<1x5x3x32x32xf32>, %arg1: tensor<5x2x3x3x3xf32>, %arg2: tensor<1x5x2x30x30xf32>) -> tensor<1x5x2x30x30xf32> {
// CHECK:     %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<1x5x3x32x32xf32>, tensor<5x2x3x3x3xf32>) outs(%arg2 : tensor<1x5x2x30x30xf32>) {
// CHECK:     ^bb0(%in: f32, %in_0: f32, %out: f32):
// CHECK:       %1 = arith.mulf %in, %in_0 : f32
// CHECK:       %2 = arith.addf %out, %1 : f32
// CHECK:       linalg.yield %2 : f32
// CHECK:     } -> tensor<1x5x2x30x30xf32>
// CHECK:     return %0 : tensor<1x5x2x30x30xf32>
// CHECK:   }
// CHECK: }
func.func @conv_2d_ngchw_gfchw(%input: tensor<1x5x3x32x32xf32>, %filter: tensor<5x2x3x3x3xf32>, %init: tensor<1x5x2x30x30xf32>) -> tensor<1x5x2x30x30xf32> {

  %0 = linalg.conv_2d_ngchw_gfchw {dilations = dense<1> : tensor<2xi64>,
                                         strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: tensor<1x5x3x32x32xf32>, tensor<5x2x3x3x3xf32>)
    outs (%init: tensor<1x5x2x30x30xf32>) -> tensor<1x5x2x30x30xf32>
  return %0 : tensor<1x5x2x30x30xf32>
}

// -----

// CHECK: #map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
// CHECK: #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>
// CHECK: #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
// CHECK: module {
// CHECK:   func.func @conv_2d_nhwc_fhwc(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<?x?x?x?xf32>, %arg2: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
// CHECK:     %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) outs(%arg2 : tensor<?x?x?x?xf32>) {
// CHECK:     ^bb0(%in: f32, %in_0: f32, %out: f32):
// CHECK:       %1 = arith.mulf %in, %in_0 : f32
// CHECK:       %2 = arith.addf %out, %1 : f32
// CHECK:       linalg.yield %2 : f32
// CHECK:     } -> tensor<?x?x?x?xf32>
// CHECK:     return %0 : tensor<?x?x?x?xf32>
// CHECK:   }
// CHECK: }
func.func @conv_2d_nhwc_fhwc(%input: tensor<?x?x?x?xf32>, %filter: tensor<?x?x?x?xf32>, %init: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {

  %0 = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>,
                                 strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
    outs (%init: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}

// -----

// CHECK: #map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
// CHECK: #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
// CHECK: #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
// CHECK: module {
// CHECK:   func.func @conv_16433136(%arg0: tensor<1x16x16x4xf32>, %arg1: tensor<3x3x4x16xf32>, %arg2: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32> {
// CHECK:     %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<1x16x16x4xf32>, tensor<3x3x4x16xf32>) outs(%arg2 : tensor<1x14x14x16xf32>) {
// CHECK:     ^bb0(%in: f32, %in_0: f32, %out: f32):
// CHECK:       %1 = arith.mulf %in, %in_0 : f32
// CHECK:       %2 = arith.addf %out, %1 : f32
// CHECK:       linalg.yield %2 : f32
// CHECK:     } -> tensor<1x14x14x16xf32>
// CHECK:     return %0 : tensor<1x14x14x16xf32>
// CHECK:   }
// CHECK: }
func.func @conv_16433136(%arg0: tensor<1x16x16x4xf32>, %arg1: tensor<3x3x4x16xf32>, %arg2: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32> {
    %0 = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
       ins(%arg0, %arg1: tensor<1x16x16x4xf32>, tensor<3x3x4x16xf32>)
      outs(%arg2: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32>
    return %0 : tensor<1x14x14x16xf32>
}

// -----

// CHECK: #map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 + d5, d2 + d6, d3, d7)>
// CHECK: #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>
// CHECK: #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>
// CHECK: module {
// CHECK:   func.func @conv_2d_nhwgc_gfhwc(%arg0: memref<?x?x?x?x?xf32>, %arg1: memref<?x?x?x?x?xf32>, %arg2: memref<?x?x?x?x?xf32>) {
// CHECK:     linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>) outs(%arg2 : memref<?x?x?x?x?xf32>) {
// CHECK:     ^bb0(%in: f32, %in_0: f32, %out: f32):
// CHECK:       %0 = arith.mulf %in, %in_0 : f32
// CHECK:       %1 = arith.addf %out, %0 : f32
// CHECK:       linalg.yield %1 : f32
// CHECK:     }
// CHECK:     return
// CHECK:   }
// CHECK: }
func.func @conv_2d_nhwgc_gfhwc(%input: memref<?x?x?x?x?xf32>, %filter: memref<?x?x?x?x?xf32>, %output: memref<?x?x?x?x?xf32>) {

  linalg.conv_2d_nhwgc_gfhwc {dilations = dense<1> : tensor<2xi64>,
                                         strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>)
    outs (%output: memref<?x?x?x?x?xf32>)
  return
}

// -----

// CHECK: #map = affine_map<(d0, d1, d2, d3) -> (d0 + d2, d1 + d3)>
// CHECK: #map1 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
// CHECK: #map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
// CHECK: module {
// CHECK:   func.func @conv(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
// CHECK:     linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%arg0, %arg1 : memref<?x?xf32>, memref<?x?xf32>) outs(%arg2 : memref<?x?xf32>) {
// CHECK:     ^bb0(%in: f32, %in_0: f32, %out: f32):
// CHECK:       %0 = arith.mulf %in, %in_0 : f32
// CHECK:       %1 = arith.addf %out, %0 : f32
// CHECK:       linalg.yield %1 : f32
// CHECK:     }
// CHECK:     return
// CHECK:   }
// CHECK: }
func.func @conv(%arg0 : memref<?x?xf32>, %arg1 : memref<?x?xf32>, %arg2 : memref<?x?xf32>) {
  linalg.conv_2d ins(%arg0, %arg1 : memref<?x?xf32>, memref<?x?xf32>) outs(%arg2 : memref<?x?xf32>)
  return
}

// -----

// CHECK: #map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 + d6, d3 + d7, d4 + d8)>
// CHECK: #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d5, d6, d7, d8)>
// CHECK: #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>
// CHECK: module {
// CHECK:   func.func @conv_3d_ncdhw_fcdhw(%arg0: tensor<?x?x?x?x?xf32>, %arg1: tensor<?x?x?x?x?xf32>, %arg2: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32> {
// CHECK:     %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>) outs(%arg2 : tensor<?x?x?x?x?xf32>) {
// CHECK:     ^bb0(%in: f32, %in_0: f32, %out: f32):
// CHECK:       %1 = arith.mulf %in, %in_0 : f32
// CHECK:       %2 = arith.addf %out, %1 : f32
// CHECK:       linalg.yield %2 : f32
// CHECK:     } -> tensor<?x?x?x?x?xf32>
// CHECK:     return %0 : tensor<?x?x?x?x?xf32>
// CHECK:   }
// CHECK: }
func.func @conv_3d_ncdhw_fcdhw(%input: tensor<?x?x?x?x?xf32>, %filter: tensor<?x?x?x?x?xf32>, %init: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32> {

  %0 = linalg.conv_3d_ncdhw_fcdhw {dilations = dense<1> : tensor<3xi64>,
                                                strides = dense<1> : tensor<3xi64>}
     ins (%input, %filter: tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>)
    outs (%init: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
  return %0 : tensor<?x?x?x?x?xf32>
}

// -----

// CHECK: #map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1 + d5, d2 + d6, d3 + d7, d8)>
// CHECK: #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d5, d6, d7, d8, d4)>
// CHECK: #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>
// CHECK: module {
// CHECK:   func.func @conv_3d_ndhwc_dhwcf(%arg0: tensor<?x?x?x?x?xf32>, %arg1: tensor<?x?x?x?x?xf32>, %arg2: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32> {
// CHECK:     %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>) outs(%arg2 : tensor<?x?x?x?x?xf32>) {
// CHECK:     ^bb0(%in: f32, %in_0: f32, %out: f32):
// CHECK:       %1 = arith.mulf %in, %in_0 : f32
// CHECK:       %2 = arith.addf %out, %1 : f32
// CHECK:       linalg.yield %2 : f32
// CHECK:     } -> tensor<?x?x?x?x?xf32>
// CHECK:     return %0 : tensor<?x?x?x?x?xf32>
// CHECK:   }
// CHECK: }
func.func @conv_3d_ndhwc_dhwcf(%input: tensor<?x?x?x?x?xf32>, %filter: tensor<?x?x?x?x?xf32>, %init: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32> {

  %0 = linalg.conv_3d_ndhwc_dhwcf {dilations = dense<1> : tensor<3xi64>,
                                                strides = dense<1> : tensor<3xi64>}
     ins (%input, %filter: tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>)
    outs (%init: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
  return %0 : tensor<?x?x?x?x?xf32>
}

// -----

// CHECK: #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 + d3, d1 + d4, d2 + d5)>
// CHECK: #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5)>
// CHECK: #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>
// CHECK: module {
// CHECK:   func.func @conv_3d(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>, %arg2: memref<?x?x?xf32>) {
// CHECK:     linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : memref<?x?x?xf32>, memref<?x?x?xf32>) outs(%arg2 : memref<?x?x?xf32>) {
// CHECK:     ^bb0(%in: f32, %in_0: f32, %out: f32):
// CHECK:       %0 = arith.mulf %in, %in_0 : f32
// CHECK:       %1 = arith.addf %out, %0 : f32
// CHECK:       linalg.yield %1 : f32
// CHECK:     }
// CHECK:     return
// CHECK:   }
// CHECK: }
func.func @conv_3d(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>, %arg2: memref<?x?x?xf32>) {
  linalg.conv_3d ins (%arg0, %arg1: memref<?x?x?xf32>, memref<?x?x?xf32>)
                outs (%arg2: memref<?x?x?xf32>)
  return
}

// -----

// CHECK: #map = affine_map<(d0, d1, d2) -> ()>
// CHECK: #map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2 + d3)>
// CHECK: #map3 = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
// CHECK: #map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK: module {
// CHECK:   func.func @depthwise_conv_1d_ncw_cw(%arg0: tensor<1x8x12xf32>, %arg1: tensor<8x3xf32>) -> tensor<1x8x10xf32> {
// CHECK:     %cst = arith.constant 0.000000e+00 : f32
// CHECK:     %0 = tensor.empty() : tensor<1x8x10xf32>
// CHECK:     %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst : f32) outs(%0 : tensor<1x8x10xf32>) {
// CHECK:     ^bb0(%in: f32, %out: f32):
// CHECK:       linalg.yield %in : f32
// CHECK:     } -> tensor<1x8x10xf32>
// CHECK:     %2 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<1x8x12xf32>, tensor<8x3xf32>) outs(%1 : tensor<1x8x10xf32>) {
// CHECK:     ^bb0(%in: f32, %in_0: f32, %out: f32):
// CHECK:       %3 = arith.mulf %in, %in_0 : f32
// CHECK:       %4 = arith.addf %out, %3 : f32
// CHECK:       linalg.yield %4 : f32
// CHECK:     } -> tensor<1x8x10xf32>
// CHECK:     return %2 : tensor<1x8x10xf32>
// CHECK:   }
// CHECK: }
func.func @depthwise_conv_1d_ncw_cw(%input: tensor<1x8x12xf32>, %filter: tensor<8x3xf32>) -> tensor<1x8x10xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<1x8x10xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%init : tensor<1x8x10xf32>) -> tensor<1x8x10xf32>

  %0 = linalg.depthwise_conv_1d_ncw_cw {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
    ins(%input, %filter : tensor<1x8x12xf32>, tensor<8x3xf32>)
    outs(%fill : tensor<1x8x10xf32>) -> tensor<1x8x10xf32>
  return %0 : tensor<1x8x10xf32>
}

// -----

// CHECK: #map = affine_map<(d0, d1, d2) -> ()>
// CHECK: #map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1 + d3, d2)>
// CHECK: #map3 = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
// CHECK: #map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK: module {
// CHECK:   func.func @depthwise_conv_1d_nwc_wc(%arg0: tensor<1x12x8xf32>, %arg1: tensor<3x8xf32>) -> tensor<1x10x8xf32> {
// CHECK:     %cst = arith.constant 0.000000e+00 : f32
// CHECK:     %0 = tensor.empty() : tensor<1x10x8xf32>
// CHECK:     %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst : f32) outs(%0 : tensor<1x10x8xf32>) {
// CHECK:     ^bb0(%in: f32, %out: f32):
// CHECK:       linalg.yield %in : f32
// CHECK:     } -> tensor<1x10x8xf32>
// CHECK:     %2 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<1x12x8xf32>, tensor<3x8xf32>) outs(%1 : tensor<1x10x8xf32>) {
// CHECK:     ^bb0(%in: f32, %in_0: f32, %out: f32):
// CHECK:       %3 = arith.mulf %in, %in_0 : f32
// CHECK:       %4 = arith.addf %out, %3 : f32
// CHECK:       linalg.yield %4 : f32
// CHECK:     } -> tensor<1x10x8xf32>
// CHECK:     return %2 : tensor<1x10x8xf32>
// CHECK:   }
// CHECK: }
func.func @depthwise_conv_1d_nwc_wc(%input: tensor<1x12x8xf32>, %filter: tensor<3x8xf32>) -> tensor<1x10x8xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<1x10x8xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%init : tensor<1x10x8xf32>) -> tensor<1x10x8xf32>

  %0 = linalg.depthwise_conv_1d_nwc_wc {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
    ins(%input, %filter : tensor<1x12x8xf32>, tensor<3x8xf32>)
    outs(%fill : tensor<1x10x8xf32>) -> tensor<1x10x8xf32>
  return %0 : tensor<1x10x8xf32>
}

// -----

// CHECK: #map = affine_map<(d0, d1, d2, d3) -> ()>
// CHECK: #map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: #map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 + d4, d2)>
// CHECK: #map3 = affine_map<(d0, d1, d2, d3, d4) -> (d4, d2, d3)>
// CHECK: #map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
// CHECK: module {
// CHECK:   func.func @depthwise_conv_1d_nwc_wcm(%arg0: tensor<1x12x8xf32>, %arg1: tensor<3x8x8xf32>) -> tensor<1x10x8x8xf32> {
// CHECK:     %cst = arith.constant 0.000000e+00 : f32
// CHECK:     %0 = tensor.empty() : tensor<1x10x8x8xf32>
// CHECK:     %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst : f32) outs(%0 : tensor<1x10x8x8xf32>) {
// CHECK:     ^bb0(%in: f32, %out: f32):
// CHECK:       linalg.yield %in : f32
// CHECK:     } -> tensor<1x10x8x8xf32>
// CHECK:     %2 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<1x12x8xf32>, tensor<3x8x8xf32>) outs(%1 : tensor<1x10x8x8xf32>) {
// CHECK:     ^bb0(%in: f32, %in_0: f32, %out: f32):
// CHECK:       %3 = arith.mulf %in, %in_0 : f32
// CHECK:       %4 = arith.addf %out, %3 : f32
// CHECK:       linalg.yield %4 : f32
// CHECK:     } -> tensor<1x10x8x8xf32>
// CHECK:     return %2 : tensor<1x10x8x8xf32>
// CHECK:   }
// CHECK: }
func.func @depthwise_conv_1d_nwc_wcm(%input: tensor<1x12x8xf32>, %filter: tensor<3x8x8xf32>) -> tensor<1x10x8x8xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<1x10x8x8xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%init : tensor<1x10x8x8xf32>) -> tensor<1x10x8x8xf32>

  %0 = linalg.depthwise_conv_1d_nwc_wcm {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
    ins(%input, %filter : tensor<1x12x8xf32>, tensor<3x8x8xf32>)
    outs(%fill : tensor<1x10x8x8xf32>) -> tensor<1x10x8x8xf32>
  return %0 : tensor<1x10x8x8xf32>
}

// -----

// CHECK: #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2 * 2 + d4, d3 * 2 + d5)>
// CHECK: #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d4, d5)>
// CHECK: #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
// CHECK: module {
// CHECK:   func.func @depthwise_conv_2d_nchw_chw_tensor(%arg0: tensor<1x96x113x113xf32>, %arg1: tensor<96x3x3xf32>) -> tensor<1x96x56x56xf32> {
// CHECK:     %0 = tensor.empty() : tensor<1x96x56x56xf32>
// CHECK:     %1 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<1x96x113x113xf32>, tensor<96x3x3xf32>) outs(%0 : tensor<1x96x56x56xf32>) {
// CHECK:     ^bb0(%in: f32, %in_0: f32, %out: f32):
// CHECK:       %2 = arith.mulf %in, %in_0 : f32
// CHECK:       %3 = arith.addf %out, %2 : f32
// CHECK:       linalg.yield %3 : f32
// CHECK:     } -> tensor<1x96x56x56xf32>
// CHECK:     return %1 : tensor<1x96x56x56xf32>
// CHECK:   }
// CHECK: }
func.func @depthwise_conv_2d_nchw_chw_tensor(%input: tensor<1x96x113x113xf32>, %filter: tensor<96x3x3xf32>) -> tensor<1x96x56x56xf32> {
  %init = tensor.empty() : tensor<1x96x56x56xf32>

  %0 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
         ins(%input, %filter: tensor<1x96x113x113xf32>, tensor<96x3x3xf32>)
         outs(%init: tensor<1x96x56x56xf32>) -> tensor<1x96x56x56xf32>
  return %0: tensor<1x96x56x56xf32>
}

// -----

// CHECK: #map = affine_map<(d0, d1, d2, d3) -> ()>
// CHECK: #map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 + d4, d2 + d5, d3)>
// CHECK: #map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5, d3)>
// CHECK: #map4 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
// CHECK: module {
// CHECK:   func.func @convolution_depthwise(%arg0: tensor<1x10x196x48xf32>, %arg1: tensor<1x4x48xf32>) -> tensor<1x10x191x48xf32> {
// CHECK:     %cst = arith.constant 0.000000e+00 : f32
// CHECK:     %0 = tensor.empty() : tensor<1x10x191x48xf32>
// CHECK:     %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst : f32) outs(%0 : tensor<1x10x191x48xf32>) {
// CHECK:     ^bb0(%in: f32, %out: f32):
// CHECK:       linalg.yield %in : f32
// CHECK:     } -> tensor<1x10x191x48xf32>
// CHECK:     %2 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<1x10x196x48xf32>, tensor<1x4x48xf32>) outs(%1 : tensor<1x10x191x48xf32>) {
// CHECK:     ^bb0(%in: f32, %in_0: f32, %out: f32):
// CHECK:       %3 = arith.mulf %in, %in_0 : f32
// CHECK:       %4 = arith.addf %out, %3 : f32
// CHECK:       linalg.yield %4 : f32
// CHECK:     } -> tensor<1x10x191x48xf32>
// CHECK:     return %2 : tensor<1x10x191x48xf32>
// CHECK:   }
// CHECK: }
func.func @convolution_depthwise(%input: tensor<1x10x196x48xf32>, %filter: tensor<1x4x48xf32>) -> tensor<1x10x191x48xf32> {
  %cst = arith.constant 0.0 : f32 
  %empty = tensor.empty() : tensor<1x10x191x48xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<1x10x191x48xf32>) -> tensor<1x10x191x48xf32>

  %result = linalg.depthwise_conv_2d_nhwc_hwc {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>}
    ins(%input, %filter : tensor<1x10x196x48xf32>, tensor<1x4x48xf32>)
    outs(%fill : tensor<1x10x191x48xf32>) -> tensor<1x10x191x48xf32>

  return %result : tensor<1x10x191x48xf32>
}

// -----

// CHECK: #map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d5, d2 + d6, d3)>
// CHECK: #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d5, d6, d3, d4)>
// CHECK: #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4)>
// CHECK: module {
// CHECK:   func.func @depthwise_conv_2d_nhwc_hwcm(%arg0: memref<2x4x5x2xf32>, %arg1: memref<2x2x2x3xf32>, %arg2: memref<2x3x4x2x3xf32>) {
// CHECK:     linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg0, %arg1 : memref<2x4x5x2xf32>, memref<2x2x2x3xf32>) outs(%arg2 : memref<2x3x4x2x3xf32>) {
// CHECK:     ^bb0(%in: f32, %in_0: f32, %out: f32):
// CHECK:       %0 = arith.mulf %in, %in_0 : f32
// CHECK:       %1 = arith.addf %out, %0 : f32
// CHECK:       linalg.yield %1 : f32
// CHECK:     }
// CHECK:     return
// CHECK:   }
// CHECK: }
func.func @depthwise_conv_2d_nhwc_hwcm(%input: memref<2x4x5x2xf32>, %filter: memref<2x2x2x3xf32>, %output: memref<2x3x4x2x3xf32>) {
  linalg.depthwise_conv_2d_nhwc_hwcm
     { dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
     ins(%input, %filter : memref<2x4x5x2xf32>, memref<2x2x2x3xf32>)
    outs(%output : memref<2x3x4x2x3xf32>)
  return
}

// -----

// CHECK: #map = affine_map<(d0, d1, d2, d3, d4) -> ()>
// CHECK: #map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK: #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2 * 2 + d5, d3 + d6, d4 * 3 + d7)>
// CHECK: #map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d1, d5, d6, d7)>
// CHECK: #map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>
// CHECK: module {
// CHECK:   func.func @depthwise_conv_3d_ncdhw_cdhw(%arg0: tensor<2x6x6x13x12xf32>, %arg1: tensor<6x2x1x3xf32>) -> tensor<2x6x3x13x4xf32> {
// CHECK:     %cst = arith.constant 0.000000e+00 : f32
// CHECK:     %0 = tensor.empty() : tensor<2x6x3x13x4xf32>
// CHECK:     %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%cst : f32) outs(%0 : tensor<2x6x3x13x4xf32>) {
// CHECK:     ^bb0(%in: f32, %out: f32):
// CHECK:       linalg.yield %in : f32
// CHECK:     } -> tensor<2x6x3x13x4xf32>
// CHECK:     %2 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<2x6x6x13x12xf32>, tensor<6x2x1x3xf32>) outs(%1 : tensor<2x6x3x13x4xf32>) {
// CHECK:     ^bb0(%in: f32, %in_0: f32, %out: f32):
// CHECK:       %3 = arith.mulf %in, %in_0 : f32
// CHECK:       %4 = arith.addf %out, %3 : f32
// CHECK:       linalg.yield %4 : f32
// CHECK:     } -> tensor<2x6x3x13x4xf32>
// CHECK:     return %2 : tensor<2x6x3x13x4xf32>
// CHECK:   }
// CHECK: }
func.func @depthwise_conv_3d_ncdhw_cdhw(%input: tensor<2x6x6x13x12xf32>, %filter: tensor<6x2x1x3xf32>) -> tensor<2x6x3x13x4xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<2x6x3x13x4xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%init : tensor<2x6x3x13x4xf32>) -> tensor<2x6x3x13x4xf32>

  %0 = linalg.depthwise_conv_3d_ncdhw_cdhw {dilations = dense<1> : tensor<3xi64>, strides = dense<[2, 1, 3]> : tensor<3xi64>}
    ins(%input, %filter : tensor<2x6x6x13x12xf32>, tensor<6x2x1x3xf32>)
    outs(%fill : tensor<2x6x3x13x4xf32>) -> tensor<2x6x3x13x4xf32>
  return %0 : tensor<2x6x3x13x4xf32>
}

// -----

// CHECK: #map = affine_map<(d0, d1, d2, d3, d4) -> ()>
// CHECK: #map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK: #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 * 2 + d5, d2 + d6, d3 * 3 + d7, d4)>
// CHECK: #map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d5, d6, d7, d4)>
// CHECK: #map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>
// CHECK: module {
// CHECK:   func.func @depthwise_conv_3d_ndhwc_dhwc(%arg0: tensor<2x6x13x12x6xf32>, %arg1: tensor<2x1x3x6xf32>) -> tensor<2x3x13x4x6xf32> {
// CHECK:     %cst = arith.constant 0.000000e+00 : f32
// CHECK:     %0 = tensor.empty() : tensor<2x3x13x4x6xf32>
// CHECK:     %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%cst : f32) outs(%0 : tensor<2x3x13x4x6xf32>) {
// CHECK:     ^bb0(%in: f32, %out: f32):
// CHECK:       linalg.yield %in : f32
// CHECK:     } -> tensor<2x3x13x4x6xf32>
// CHECK:     %2 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<2x6x13x12x6xf32>, tensor<2x1x3x6xf32>) outs(%1 : tensor<2x3x13x4x6xf32>) {
// CHECK:     ^bb0(%in: f32, %in_0: f32, %out: f32):
// CHECK:       %3 = arith.mulf %in, %in_0 : f32
// CHECK:       %4 = arith.addf %out, %3 : f32
// CHECK:       linalg.yield %4 : f32
// CHECK:     } -> tensor<2x3x13x4x6xf32>
// CHECK:     return %2 : tensor<2x3x13x4x6xf32>
// CHECK:   }
// CHECK: }
func.func @depthwise_conv_3d_ndhwc_dhwc(%input: tensor<2x6x13x12x6xf32>, %filter: tensor<2x1x3x6xf32>) -> tensor<2x3x13x4x6xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<2x3x13x4x6xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%init : tensor<2x3x13x4x6xf32>) -> tensor<2x3x13x4x6xf32>

  %0 = linalg.depthwise_conv_3d_ndhwc_dhwc {dilations = dense<1> : tensor<3xi64>, strides = dense<[2, 1, 3]> : tensor<3xi64>}
    ins(%input, %filter : tensor<2x6x13x12x6xf32>, tensor<2x1x3x6xf32>)
    outs(%fill : tensor<2x3x13x4x6xf32>) -> tensor<2x3x13x4x6xf32>
  return %0 : tensor<2x3x13x4x6xf32>
}

// -----

// CHECK: #map = affine_map<(d0, d1, d2, d3, d4, d5) -> ()>
// CHECK: #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
// CHECK: #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1 * 2 + d6, d2 + d7, d3 * 3 + d8, d4)>
// CHECK: #map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d6, d7, d8, d4, d5)>
// CHECK: #map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4, d5)>
// CHECK: module {
// CHECK:   func.func @depthwise_conv_3d_ndhwc_dhwcm(%arg0: tensor<2x6x13x12x6xf32>, %arg1: tensor<2x1x3x6x6xf32>) -> tensor<2x3x13x4x6x6xf32> {
// CHECK:     %cst = arith.constant 0.000000e+00 : f32
// CHECK:     %0 = tensor.empty() : tensor<2x3x13x4x6x6xf32>
// CHECK:     %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%cst : f32) outs(%0 : tensor<2x3x13x4x6x6xf32>) {
// CHECK:     ^bb0(%in: f32, %out: f32):
// CHECK:       linalg.yield %in : f32
// CHECK:     } -> tensor<2x3x13x4x6x6xf32>
// CHECK:     %2 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<2x6x13x12x6xf32>, tensor<2x1x3x6x6xf32>) outs(%1 : tensor<2x3x13x4x6x6xf32>) {
// CHECK:     ^bb0(%in: f32, %in_0: f32, %out: f32):
// CHECK:       %3 = arith.mulf %in, %in_0 : f32
// CHECK:       %4 = arith.addf %out, %3 : f32
// CHECK:       linalg.yield %4 : f32
// CHECK:     } -> tensor<2x3x13x4x6x6xf32>
// CHECK:     return %2 : tensor<2x3x13x4x6x6xf32>
// CHECK:   }
// CHECK: }
func.func @depthwise_conv_3d_ndhwc_dhwcm(%input: tensor<2x6x13x12x6xf32>, %filter: tensor<2x1x3x6x6xf32>) -> tensor<2x3x13x4x6x6xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<2x3x13x4x6x6xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%init : tensor<2x3x13x4x6x6xf32>) -> tensor<2x3x13x4x6x6xf32>

  %0 = linalg.depthwise_conv_3d_ndhwc_dhwcm {dilations = dense<1> : tensor<3xi64>, strides = dense<[2, 1, 3]> : tensor<3xi64>}
    ins(%input, %filter : tensor<2x6x13x12x6xf32>, tensor<2x1x3x6x6xf32>)
    outs(%fill : tensor<2x3x13x4x6x6xf32>) -> tensor<2x3x13x4x6x6xf32>
  return %0 : tensor<2x3x13x4x6x6xf32>
}
