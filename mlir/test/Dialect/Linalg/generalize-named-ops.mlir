// RUN: mlir-opt %s -split-input-file -linalg-generalize-named-ops | FileCheck %s

func @generalize_matmul_buffer(%A : memref<16x8xf32>, %B: memref<8x32xf32>, %C: memref<16x32xf32>) {
  linalg.matmul ins(%A, %B: memref<16x8xf32>, memref<8x32xf32>)
               outs(%C: memref<16x32xf32>)
  return
}


// CHECK: #[[A_MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[B_MAP:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK: #[[C_MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK: func @generalize_matmul_buffer
// CHECK-SAME: %[[A:.+]]: memref<16x8xf32>
// CHECK-SAME: %[[B:.+]]: memref<8x32xf32>
// CHECK-SAME: %[[C:.+]]: memref<16x32xf32>

// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[A_MAP]], #[[B_MAP]], #[[C_MAP]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-SAME:  ins(%[[A]], %[[B]]
// CHECK-SAME: outs(%[[C]]

// CHECK: ^{{.*}}(%[[A_ARG:.+]]: f32, %[[B_ARG:.+]]: f32, %[[C_ARG:.+]]: f32)
// CHECK:   %[[MUL:.+]] = arith.mulf %[[A_ARG]], %[[B_ARG]] : f32
// CHECK:   %[[ADD:.+]] = arith.addf %[[C_ARG]], %[[MUL]] : f32
// CHECK:   linalg.yield %[[ADD]] : f32

// -----

func @generalize_matmul_tensor(%A : tensor<16x8xf32>, %B: tensor<8x32xf32>, %C: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = linalg.matmul ins(%A, %B: tensor<16x8xf32>, tensor<8x32xf32>)
                    outs(%C: tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0: tensor<16x32xf32>
}

// CHECK: func @generalize_matmul_tensor

// CHECK: linalg.generic
// CHECK-SAME:  ins(%{{.+}}, %{{.+}} : tensor<16x8xf32>, tensor<8x32xf32>)
// CHECK-SAME: outs(%{{.+}} : tensor<16x32xf32>)

// CHECK:      ^{{.*}}(%[[A_ARG:.+]]: f32, %[[B_ARG:.+]]: f32, %[[C_ARG:.+]]: f32)
// CHECK-NEXT:   %[[MUL:.+]] = arith.mulf %[[A_ARG]], %[[B_ARG]] : f32
// CHECK-NEXT:   %[[ADD:.+]] = arith.addf %[[C_ARG]], %[[MUL]] : f32
// CHECK-NEXT:   linalg.yield %[[ADD]] : f32
// CHECK-NEXT: -> tensor<16x32xf32>

// -----

func @depthwise_conv_2d_nhwc_hwcm(%input: memref<2x4x5x2xf32>, %filter: memref<2x2x2x3xf32>, %output: memref<2x3x4x2x3xf32>) {
  linalg.depthwise_conv_2d_nhwc_hwcm
     { dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
     ins(%input, %filter : memref<2x4x5x2xf32>, memref<2x2x2x3xf32>)
    outs(%output : memref<2x3x4x2x3xf32>)
  return
}

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d5, d2 + d6, d3)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d5, d6, d3, d4)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4)>

// CHECK: func @depthwise_conv_2d_nhwc_hwcm

// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]}
// CHECK-SAME: ins(%{{.+}}, %{{.+}} : memref<2x4x5x2xf32>, memref<2x2x2x3xf32>)
// CHECK-SAME: outs(%{{.+}} : memref<2x3x4x2x3xf32>)

// CHECK:         ^{{.+}}(%[[BBARG0:.+]]: f32, %[[BBARG1:.+]]: f32, %[[BBARG2:.+]]: f32)
// CHECK-NEXT:      %[[MUL:.+]] = arith.mulf %[[BBARG0]], %[[BBARG1]] : f32
// CHECK-NEXT:      %[[ADD:.+]] = arith.addf %[[BBARG2]], %[[MUL]] : f32
// CHECK-NEXT:      linalg.yield %[[ADD]] : f32

// -----

func @depthwise_conv_2d_nhwc_hwcm(%input: memref<2x4x5x2xf32>, %filter: memref<2x2x2x3xf32>, %output: memref<2x2x3x2x3xf32>) {
  linalg.depthwise_conv_2d_nhwc_hwcm
     { dilations = dense<2> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
     ins(%input, %filter : memref<2x4x5x2xf32>, memref<2x2x2x3xf32>)
    outs(%output : memref<2x2x3x2x3xf32>)
  return
}

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d5 * 2, d2 + d6 * 2, d3)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d5, d6, d3, d4)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4)>

// CHECK: func @depthwise_conv_2d_nhwc_hwcm

// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]}
// CHECK-SAME: ins(%{{.+}}, %{{.+}} : memref<2x4x5x2xf32>, memref<2x2x2x3xf32>)
// CHECK-SAME: outs(%{{.+}} : memref<2x2x3x2x3xf32>)

// CHECK:         ^{{.+}}(%[[BBARG0:.+]]: f32, %[[BBARG1:.+]]: f32, %[[BBARG2:.+]]: f32)
// CHECK-NEXT:      %[[MUL:.+]] = arith.mulf %[[BBARG0]], %[[BBARG1]] : f32
// CHECK-NEXT:      %[[ADD:.+]] = arith.addf %[[BBARG2]], %[[MUL]] : f32
// CHECK-NEXT:      linalg.yield %[[ADD]] : f32

// -----

func @depthwise_conv_2d_nhwc_hwc(%input: memref<1x113x113x96xf32>, %filter: memref<3x3x96xf32>, %output: memref<1x56x56x96xf32>) {
  linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
    ins(%input, %filter: memref<1x113x113x96xf32>, memref<3x3x96xf32>)
    outs(%output: memref<1x56x56x96xf32>)
  return
}

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 * 2 + d4, d2 * 2 + d5, d3)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5, d3)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>

// CHECK: func @depthwise_conv_2d_nhwc_hwc

// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]}
// CHECK-SAME: ins(%{{.+}}, %{{.+}} : memref<1x113x113x96xf32>, memref<3x3x96xf32>)
// CHECK-SAME: outs(%{{.+}} : memref<1x56x56x96xf32>)

// CHECK:         ^{{.+}}(%[[BBARG0:.+]]: f32, %[[BBARG1:.+]]: f32, %[[BBARG2:.+]]: f32)
// CHECK-NEXT:      %[[MUL:.+]] = arith.mulf %[[BBARG0]], %[[BBARG1]] : f32
// CHECK-NEXT:      %[[ADD:.+]] = arith.addf %[[BBARG2]], %[[MUL]] : f32
// CHECK-NEXT:      linalg.yield %[[ADD]] : f32

// -----

func @conv_1d_nwc_wcf(%input: memref<?x?x?xf32>, %filter: memref<?x?x?xf32>, %output: memref<?x?x?xf32>) {
  linalg.conv_1d_nwc_wcf {dilations = dense<1> : tensor<1xi64>,
                                       strides = dense<1> : tensor<1xi64>}
     ins (%input, %filter: memref<?x?x?xf32>, memref<?x?x?xf32>)
    outs (%output: memref<?x?x?xf32>)
  return
}
// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 + d3, d4)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d3, d4, d2)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>

// CHECK: func @conv_1d_nwc_wcf

// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]}
// CHECK-SAME: ins(%{{.+}}, %{{.+}} : memref<?x?x?xf32>, memref<?x?x?xf32>)
// CHECK-SAME: outs(%{{.+}} : memref<?x?x?xf32>)

// CHECK:         ^{{.+}}(%[[BBARG0:.+]]: f32, %[[BBARG1:.+]]: f32, %[[BBARG2:.+]]: f32)
// CHECK-NEXT:      %[[MUL:.+]] = arith.mulf %[[BBARG0]], %[[BBARG1]] : f32
// CHECK-NEXT:      %[[ADD:.+]] = arith.addf %[[BBARG2]], %[[MUL]] : f32
// CHECK-NEXT:      linalg.yield %[[ADD]] : f32

// -----

func @generalize_fill(%output: memref<?x?xf32>, %value : f32) {
  linalg.fill ins(%value : f32) outs(%output : memref<?x?xf32>)
  return
}

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1) -> ()>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK: func @generalize_fill
// CHECK-SAME: (%[[ARG0:.+]]: memref<?x?xf32>, %[[VAL:.+]]: f32)

// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel"]}
// CHECK-SAME: ins(%[[VAL]] : f32)
// CHECK-SAME: outs(%{{.+}} : memref<?x?xf32>)

// CHECK:         ^{{.+}}(%[[BBARG0:.+]]: f32, %[[BBARG1:.+]]: f32)
// CHECK-NEXT:      linalg.yield %[[BBARG0]] : f32

// -----

func @generalize_batch_matm_vec(%lhs : memref<?x?x?xi8>, %rhs: memref<?x?xi8>,  %out: memref<?x?xf32>) {
  linalg.batch_matvec ins(%lhs, %rhs: memref<?x?x?xi8>, memref<?x?xi8>)
                     outs(%out: memref<?x?xf32>)
  return
}
// CHECK: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK: @generalize_batch_matm_vec

// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]}
// CHECK-SAME: ins(%{{.+}}, %{{.+}} : memref<?x?x?xi8>, memref<?x?xi8>)
// CHECK-SAME: outs(%{{.+}} : memref<?x?xf32>)
// CHECK:         ^{{.+}}(%[[BBARG0:.+]]: i8, %[[BBARG1:.+]]: i8, %[[BBARG2:.+]]: f32)
// CHECK:            %[[BBARG0_F32:.+]] = arith.sitofp %[[BBARG0]] : i8 to f32
// CHECK:            %[[BBARG1_F32:.+]] = arith.sitofp %[[BBARG1]] : i8 to f32
// CHECK:            %[[MUL:.+]] = arith.mulf %[[BBARG0_F32]], %[[BBARG1_F32]]
// CHECK:            %[[ADD:.+]] = arith.addf %[[BBARG2]], %[[MUL]]
// CHECK:            linalg.yield %[[ADD]] : f32
