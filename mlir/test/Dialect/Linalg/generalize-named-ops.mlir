// RUN: mlir-opt %s -split-input-file -linalg-generalize-named-ops | FileCheck %s

func.func @generalize_matmul_buffer(%A : memref<16x8xf32>, %B: memref<8x32xf32>, %C: memref<16x32xf32>) {
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

func.func @matmul_bcast_a(%arg0: memref<5xf32>, %arg1: memref<5x7xf32>, %arg2: memref<3x7xf32>) {
  linalg.matmul indexing_maps = [
                       affine_map<(d0, d1, d2) -> (d2)>,
                       affine_map<(d0, d1, d2) -> (d2, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>
                     ]
                     ins(%arg0, %arg1 : memref<5xf32>, memref<5x7xf32>) outs(%arg2: memref<3x7xf32>)
  return
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL:   func.func @matmul_bcast_a(
// CHECK-SAME:                              %[[VAL_0:.*]]: memref<5xf32>,
// CHECK-SAME:                              %[[VAL_1:.*]]: memref<5x7xf32>,
// CHECK-SAME:                              %[[VAL_2:.*]]: memref<3x7xf32>) {
// CHECK:           linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel", "reduction"]} ins(%[[VAL_0]], %[[VAL_1]] : memref<5xf32>, memref<5x7xf32>) outs(%[[VAL_2]] : memref<3x7xf32>) {
// CHECK:           ^bb0(%[[VAL_3:.*]]: f32, %[[VAL_4:.*]]: f32, %[[VAL_5:.*]]: f32):
// CHECK:             %[[VAL_6:.*]] = arith.mulf %[[VAL_3]], %[[VAL_4]] : f32
// CHECK:             %[[VAL_7:.*]] = arith.addf %[[VAL_5]], %[[VAL_6]] : f32
// CHECK:             linalg.yield %[[VAL_7]] : f32
// CHECK:           }
// CHECK:           return
// CHECK:         }

// -----

func.func @generalize_matmul_tensor(%A : tensor<16x8xf32>, %B: tensor<8x32xf32>, %C: tensor<16x32xf32>) -> tensor<16x32xf32> {
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

func.func @generalize_matmul_tensor_complex(%A : tensor<16x8xcomplex<f32>>,
                                            %B: tensor<8x32xcomplex<f32>>,
                                            %C: tensor<16x32xcomplex<f32>>)
          -> tensor<16x32xcomplex<f32>> {
  %0 = linalg.matmul ins(%A, %B: tensor<16x8xcomplex<f32>>, tensor<8x32xcomplex<f32>>)
                    outs(%C: tensor<16x32xcomplex<f32>>) -> tensor<16x32xcomplex<f32>>
  return %0: tensor<16x32xcomplex<f32>>
}

// CHECK: func @generalize_matmul_tensor_complex

// CHECK: linalg.generic
// CHECK-SAME:  ins(%{{.+}}, %{{.+}} : tensor<16x8xcomplex<f32>>, tensor<8x32xcomplex<f32>>)
// CHECK-SAME: outs(%{{.+}} : tensor<16x32xcomplex<f32>>)

// CHECK:      ^{{.*}}(%[[A_ARG:.+]]: complex<f32>, %[[B_ARG:.+]]: complex<f32>, %[[C_ARG:.+]]: complex<f32>)
// CHECK-NEXT:   %[[MUL:.+]] = complex.mul %[[A_ARG]], %[[B_ARG]] : complex<f32>
// CHECK-NEXT:   %[[ADD:.+]] = complex.add %[[C_ARG]], %[[MUL]] : complex<f32>
// CHECK-NEXT:   linalg.yield %[[ADD]] : complex<f32>
// CHECK-NEXT: -> tensor<16x32xcomplex<f32>>

// -----

func.func @depthwise_conv_2d_nhwc_hwcm(%input: memref<2x4x5x2xf32>, %filter: memref<2x2x2x3xf32>, %output: memref<2x3x4x2x3xf32>) {
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

func.func @depthwise_conv_2d_nhwc_hwcm(%input: memref<2x4x5x2xf32>, %filter: memref<2x2x2x3xf32>, %output: memref<2x2x3x2x3xf32>) {
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

func.func @depthwise_conv_2d_nhwc_hwc(%input: memref<1x113x113x96xf32>, %filter: memref<3x3x96xf32>, %output: memref<1x56x56x96xf32>) {
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

func.func @conv_1d_nwc_wcf(%input: memref<?x?x?xf32>, %filter: memref<?x?x?xf32>, %output: memref<?x?x?xf32>) {
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

func.func @conv_1d_ncw_fcw(%input: memref<?x?x?xf32>, %filter: memref<?x?x?xf32>, %output: memref<?x?x?xf32>) {
  linalg.conv_1d_ncw_fcw {dilations = dense<1> : tensor<1xi64>,
                                       strides = dense<1> : tensor<1xi64>}
     ins (%input, %filter: memref<?x?x?xf32>, memref<?x?x?xf32>)
    outs (%output: memref<?x?x?xf32>)
  return
}
// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2 + d4)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>

// CHECK: func @conv_1d_ncw_fcw

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

func.func @conv_2d_ngchw_gfchw_q(%input: memref<?x?x?x?x?xi8>, %filter: memref<?x?x?x?x?xi8>, %inputzp: i32, %filterzp: i32, %output: memref<?x?x?x?x?xi32>) {
  linalg.conv_2d_ngchw_gfchw_q {dilations = dense<1> : tensor<2xi64>,
                                       strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter, %inputzp, %filterzp: memref<?x?x?x?x?xi8>, memref<?x?x?x?x?xi8>, i32, i32)
    outs (%output: memref<?x?x?x?x?xi32>)
  return
}
// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d5, d3 + d6, d4 + d7)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d1, d2, d5, d6, d7)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> ()>
// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>

// CHECK: func @conv_2d_ngchw_gfchw_q

// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]], #[[MAP2]], #[[MAP3]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]}
// CHECK-SAME: ins(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} : memref<?x?x?x?x?xi8>, memref<?x?x?x?x?xi8>, i32, i32)
// CHECK-SAME: outs(%{{.+}} : memref<?x?x?x?x?xi32>)

// CHECK:         ^{{.+}}(%[[BBARG0:.+]]: i8, %[[BBARG1:.+]]: i8, %[[BBARG2:.+]]: i32, %[[BBARG3:.+]]: i32, %[[BBARG4:.+]]: i32)
// CHECK-NEXT:      %[[EXTSI0:.+]] = arith.extsi %[[BBARG0]] : i8 to i32
// CHECK-NEXT:      %[[SUB0:.+]] = arith.subi %[[EXTSI0]], %[[BBARG2]] : i32
// CHECK-NEXT:      %[[EXTSI1:.+]] = arith.extsi %[[BBARG1]] : i8 to i32
// CHECK-NEXT:      %[[SUB1:.+]] = arith.subi %[[EXTSI1]], %[[BBARG3]] : i32
// CHECK-NEXT:      %[[MUL:.+]] = arith.muli %[[SUB0]], %[[SUB1]] : i32
// CHECK-NEXT:      %[[ADD:.+]] = arith.addi %[[BBARG4]], %[[MUL]] : i32
// CHECK-NEXT:      linalg.yield %[[ADD]] : i32

// -----

func.func @generalize_fill(%output: memref<?x?xf32>, %value : f32) {
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

func.func @generalize_batch_matm_vec(%lhs : memref<?x?x?xi8>, %rhs: memref<?x?xi8>,  %out: memref<?x?xf32>) {
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

// -----

func.func @generalize_batch_vecmat(%lhs : memref<?x?xi8>, %rhs: memref<?x?x?xi8>,  %out: memref<?x?xf32>) {
  linalg.batch_vecmat ins(%lhs, %rhs: memref<?x?xi8>, memref<?x?x?xi8>)
                     outs(%out: memref<?x?xf32>)
  return
}
// CHECK: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
// CHECK: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK: @generalize_batch_vecmat

// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]}
// CHECK-SAME: ins(%{{.+}}, %{{.+}} : memref<?x?xi8>, memref<?x?x?xi8>)
// CHECK-SAME: outs(%{{.+}} : memref<?x?xf32>)
// CHECK:         ^{{.+}}(%[[BBARG0:.+]]: i8, %[[BBARG1:.+]]: i8, %[[BBARG2:.+]]: f32)
// CHECK:            %[[BBARG0_F32:.+]] = arith.sitofp %[[BBARG0]] : i8 to f32
// CHECK:            %[[BBARG1_F32:.+]] = arith.sitofp %[[BBARG1]] : i8 to f32
// CHECK:            %[[MUL:.+]] = arith.mulf %[[BBARG0_F32]], %[[BBARG1_F32]]
// CHECK:            %[[ADD:.+]] = arith.addf %[[BBARG2]], %[[MUL]]
// CHECK:            linalg.yield %[[ADD]] : f32

// -----

func.func @batch_reduce_gemm(%lhs: memref<7x8x9xf32>, %rhs: memref<7x9x8xf32>, %out: memref<8x8xf32>) {
  linalg.batch_reduce_matmul ins(%lhs, %rhs: memref<7x8x9xf32>, memref<7x9x8xf32>)
                             outs(%out: memref<8x8xf32>)
  return
}

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2)>

// CHECK: @batch_reduce_gemm

// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME: iterator_types = ["reduction", "parallel", "parallel", "reduction"]}
// CHECK-SAME: ins(%{{.+}}, %{{.+}} : memref<7x8x9xf32>, memref<7x9x8xf32>)
// CHECK-SAME: outs(%{{.+}} : memref<8x8xf32>
// CHECK:         ^{{.+}}(%[[BBARG0:.+]]: f32, %[[BBARG1:.+]]: f32, %[[BBARG2:.+]]: f32)
// CHECK:         %[[MUL:.+]] = arith.mulf %[[BBARG0]], %[[BBARG1]] : f32
// CHECK:         %[[ADD:.+]] = arith.addf %[[BBARG2]], %[[MUL]] : f32
// CHECK:         linalg.yield %[[ADD]] : f32

// -----

func.func @generalize_batch_reduce_gemm_bf16(%lhs: memref<7x8x9xbf16>, %rhs: memref<7x9x8xbf16>, %out: memref<8x8xf32>) {
  linalg.batch_reduce_matmul ins(%lhs, %rhs: memref<7x8x9xbf16>, memref<7x9x8xbf16>)
                             outs(%out: memref<8x8xf32>)
  return
}

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2)>

// CHECK: @generalize_batch_reduce_gemm_bf16

// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME: iterator_types = ["reduction", "parallel", "parallel", "reduction"]}
// CHECK-SAME: ins(%{{.+}}, %{{.+}} : memref<7x8x9xbf16>, memref<7x9x8xbf16>)
// CHECK-SAME: outs(%{{.+}} : memref<8x8xf32>
// CHECK:         ^{{.+}}(%[[BBARG0:.+]]: bf16, %[[BBARG1:.+]]: bf16, %[[BBARG2:.+]]: f32)
// CHECK:         %[[EXTBF16_0:.+]] = arith.extf %[[BBARG0]] : bf16 to f32
// CHECK:         %[[EXTBF16_1:.+]] = arith.extf %[[BBARG1]] : bf16 to f32
// CHECK:         %[[MUL:.+]] = arith.mulf %[[EXTBF16_0]], %[[EXTBF16_1]] : f32
// CHECK:         %[[ADD:.+]] = arith.addf %[[BBARG2]], %[[MUL]] : f32
// CHECK:         linalg.yield %[[ADD]] : f32


// -----

// CHECK-LABEL: generalize_linalg_map
func.func @generalize_linalg_map(%arg0: memref<1x8x8x8xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  // CHECK: linalg.map
  // CHECK-NOT: linalg.generic
  linalg.map outs(%arg0 : memref<1x8x8x8xf32>)
    () {
      linalg.yield %cst : f32
    }
  return
}

// -----

func.func @generalize_add(%lhs: memref<7x14x21xf32>, %rhs: memref<7x14x21xf32>,
                          %out: memref<7x14x21xf32>) {
  linalg.add ins(%lhs, %rhs : memref<7x14x21xf32>, memref<7x14x21xf32>)
             outs(%out : memref<7x14x21xf32>)
  return
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK: func @generalize_add
// CHECK-SAME: (%[[LHS:.+]]: memref<7x14x21xf32>, %[[RHS:.+]]: memref<7x14x21xf32>,
// CHECK-SAME:  %[[OUT:.+]]: memref<7x14x21xf32>)

// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP]], #[[MAP]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]}
// CHECK-SAME:  ins(%[[LHS]], %[[RHS]] : memref<7x14x21xf32>, memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)

// CHECK:         ^{{.+}}(%[[BBARG0:.+]]: f32, %[[BBARG1:.+]]: f32, %[[BBARG2:.+]]: f32)
// CHECK-NEXT:      %[[SUM:.+]] = arith.addf %[[BBARG0]], %[[BBARG1]] : f32
// CHECK-NEXT:      linalg.yield %[[SUM]] : f32

// -----

func.func @generalize_sub(%lhs: memref<7x14x21xf32>, %rhs: memref<7x14x21xf32>,
                          %out: memref<7x14x21xf32>) {
  linalg.sub ins(%lhs, %rhs : memref<7x14x21xf32>, memref<7x14x21xf32>)
             outs(%out : memref<7x14x21xf32>)
  return
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK: func @generalize_sub
// CHECK-SAME: (%[[LHS:.+]]: memref<7x14x21xf32>, %[[RHS:.+]]: memref<7x14x21xf32>,
// CHECK-SAME:  %[[OUT:.+]]: memref<7x14x21xf32>)

// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP]], #[[MAP]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]}
// CHECK-SAME:  ins(%[[LHS]], %[[RHS]] : memref<7x14x21xf32>, memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)

// CHECK:         ^{{.+}}(%[[BBARG0:.+]]: f32, %[[BBARG1:.+]]: f32, %[[BBARG2:.+]]: f32)
// CHECK-NEXT:      %[[SUB:.+]] = arith.subf %[[BBARG0]], %[[BBARG1]] : f32
// CHECK-NEXT:      linalg.yield %[[SUB]] : f32

// -----

func.func @generalize_mul(%lhs: memref<7x14x21xf32>, %rhs: memref<7x14x21xf32>,
                          %out: memref<7x14x21xf32>) {
  linalg.mul ins(%lhs, %rhs : memref<7x14x21xf32>, memref<7x14x21xf32>)
             outs(%out : memref<7x14x21xf32>)
  return
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK: func @generalize_mul
// CHECK-SAME: (%[[LHS:.+]]: memref<7x14x21xf32>, %[[RHS:.+]]: memref<7x14x21xf32>,
// CHECK-SAME:  %[[OUT:.+]]: memref<7x14x21xf32>)

// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP]], #[[MAP]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]}
// CHECK-SAME:  ins(%[[LHS]], %[[RHS]] : memref<7x14x21xf32>, memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)

// CHECK:         ^{{.+}}(%[[BBARG0:.+]]: f32, %[[BBARG1:.+]]: f32, %[[BBARG2:.+]]: f32)
// CHECK-NEXT:      %[[MUL:.+]] = arith.mulf %[[BBARG0]], %[[BBARG1]] : f32
// CHECK-NEXT:      linalg.yield %[[MUL]] : f32

// -----

func.func @generalize_div(%lhs: memref<7x14x21xf32>, %rhs: memref<7x14x21xf32>,
                          %out: memref<7x14x21xf32>) {
  linalg.div ins(%lhs, %rhs : memref<7x14x21xf32>, memref<7x14x21xf32>)
             outs(%out : memref<7x14x21xf32>)
  return
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK: func @generalize_div
// CHECK-SAME: (%[[LHS:.+]]: memref<7x14x21xf32>, %[[RHS:.+]]: memref<7x14x21xf32>,
// CHECK-SAME:  %[[OUT:.+]]: memref<7x14x21xf32>)

// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP]], #[[MAP]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]}
// CHECK-SAME:  ins(%[[LHS]], %[[RHS]] : memref<7x14x21xf32>, memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)

// CHECK:         ^{{.+}}(%[[BBARG0:.+]]: f32, %[[BBARG1:.+]]: f32, %[[BBARG2:.+]]: f32)
// CHECK-NEXT:      %[[DIV:.+]] = arith.divf %[[BBARG0]], %[[BBARG1]] : f32
// CHECK-NEXT:      linalg.yield %[[DIV]] : f32

// -----

func.func @generalize_divu(%lhs: memref<7x14x21xi32>, %rhs: memref<7x14x21xi32>,
                          %out: memref<7x14x21xi32>) {
  linalg.div_unsigned ins(%lhs, %rhs : memref<7x14x21xi32>, memref<7x14x21xi32>)
             outs(%out : memref<7x14x21xi32>)
  return
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK: func @generalize_divu
// CHECK-SAME: (%[[LHS:.+]]: memref<7x14x21xi32>, %[[RHS:.+]]: memref<7x14x21xi32>,
// CHECK-SAME:  %[[OUT:.+]]: memref<7x14x21xi32>)

// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP]], #[[MAP]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]}
// CHECK-SAME:  ins(%[[LHS]], %[[RHS]] : memref<7x14x21xi32>, memref<7x14x21xi32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xi32>)

// CHECK:         ^{{.+}}(%[[BBARG0:.+]]: i32, %[[BBARG1:.+]]: i32, %[[BBARG2:.+]]: i32)
// CHECK-NEXT:      %[[DIVU:.+]] = arith.divui %[[BBARG0]], %[[BBARG1]] : i32
// CHECK-NEXT:      linalg.yield %[[DIVU]] : i32

// -----

func.func @generalize_exp(%arg: memref<7x14x21xf32>, %out: memref<7x14x21xf32>) {
  linalg.exp ins(%arg : memref<7x14x21xf32>) outs(%out : memref<7x14x21xf32>)
  return
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK: func @generalize_exp
// CHECK-SAME: (%[[ARG:.+]]: memref<7x14x21xf32>, %[[OUT:.+]]: memref<7x14x21xf32>)

// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]}
// CHECK-SAME:  ins(%[[LHS]] : memref<7x14x21xf32>) outs(%[[OUT]] : memref<7x14x21xf32>)

// CHECK:         ^{{.+}}(%[[BBARG0:.+]]: f32, %[[BBARG1:.+]]: f32)
// CHECK-NEXT:      %[[EXP:.+]] = math.exp %[[BBARG0]] : f32
// CHECK-NEXT:      linalg.yield %[[EXP]] : f32

// -----

func.func @generalize_log(%arg: memref<7x14x21xf32>, %out: memref<7x14x21xf32>) {
  linalg.log ins(%arg : memref<7x14x21xf32>) outs(%out : memref<7x14x21xf32>)
  return
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK: func @generalize_log
// CHECK-SAME: (%[[ARG:.+]]: memref<7x14x21xf32>, %[[OUT:.+]]: memref<7x14x21xf32>)

// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]}
// CHECK-SAME:  ins(%[[LHS]] : memref<7x14x21xf32>) outs(%[[OUT]] : memref<7x14x21xf32>)

// CHECK:         ^{{.+}}(%[[BBARG0:.+]]: f32, %[[BBARG1:.+]]: f32)
// CHECK-NEXT:      %[[log:.+]] = math.log %[[BBARG0]] : f32
// CHECK-NEXT:      linalg.yield %[[log]] : f32

// -----

func.func @generalize_abs(%arg: memref<7x14x21xf32>, %out: memref<7x14x21xf32>) {
  linalg.abs ins(%arg : memref<7x14x21xf32>) outs(%out : memref<7x14x21xf32>)
  return
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK: func @generalize_abs
// CHECK-SAME: (%[[ARG:.+]]: memref<7x14x21xf32>, %[[OUT:.+]]: memref<7x14x21xf32>)

// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]}
// CHECK-SAME:  ins(%[[LHS]] : memref<7x14x21xf32>) outs(%[[OUT]] : memref<7x14x21xf32>)

// CHECK:         ^{{.+}}(%[[BBARG0:.+]]: f32, %[[BBARG1:.+]]: f32)
// CHECK-NEXT:      %[[abs:.+]] = math.absf %[[BBARG0]] : f32
// CHECK-NEXT:      linalg.yield %[[abs]] : f32

// -----

func.func @generalize_ceil(%arg: memref<7x14x21xf32>, %out: memref<7x14x21xf32>) {
  linalg.ceil ins(%arg : memref<7x14x21xf32>) outs(%out : memref<7x14x21xf32>)
  return
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK: func @generalize_ceil
// CHECK-SAME: (%[[ARG:.+]]: memref<7x14x21xf32>, %[[OUT:.+]]: memref<7x14x21xf32>)

// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]}
// CHECK-SAME:  ins(%[[LHS]] : memref<7x14x21xf32>) outs(%[[OUT]] : memref<7x14x21xf32>)

// CHECK:         ^{{.+}}(%[[BBARG0:.+]]: f32, %[[BBARG1:.+]]: f32)
// CHECK-NEXT:      %[[ceil:.+]] = math.ceil %[[BBARG0]] : f32
// CHECK-NEXT:      linalg.yield %[[ceil]] : f32

// -----

func.func @generalize_floor(%arg: memref<7x14x21xf32>, %out: memref<7x14x21xf32>) {
  linalg.floor ins(%arg : memref<7x14x21xf32>) outs(%out : memref<7x14x21xf32>)
  return
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK: func @generalize_floor
// CHECK-SAME: (%[[ARG:.+]]: memref<7x14x21xf32>, %[[OUT:.+]]: memref<7x14x21xf32>)

// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]}
// CHECK-SAME:  ins(%[[LHS]] : memref<7x14x21xf32>) outs(%[[OUT]] : memref<7x14x21xf32>)

// CHECK:         ^{{.+}}(%[[BBARG0:.+]]: f32, %[[BBARG1:.+]]: f32)
// CHECK-NEXT:      %[[floor:.+]] = math.floor %[[BBARG0]] : f32
// CHECK-NEXT:      linalg.yield %[[floor]] : f32

// -----

func.func @generalize_negf(%arg: memref<7x14x21xf32>, %out: memref<7x14x21xf32>) {
  linalg.negf ins(%arg : memref<7x14x21xf32>) outs(%out : memref<7x14x21xf32>)
  return
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK: func @generalize_negf
// CHECK-SAME: (%[[ARG:.+]]: memref<7x14x21xf32>, %[[OUT:.+]]: memref<7x14x21xf32>)

// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]}
// CHECK-SAME:  ins(%[[LHS]] : memref<7x14x21xf32>) outs(%[[OUT]] : memref<7x14x21xf32>)

// CHECK:         ^{{.+}}(%[[BBARG0:.+]]: f32, %[[BBARG1:.+]]: f32)
// CHECK-NEXT:      %[[negf:.+]] = arith.negf %[[BBARG0]] : f32
// CHECK-NEXT:      linalg.yield %[[negf]] : f32

// -----

func.func @generalize_reciprocal(%arg: memref<7x14x21xf32>, %out: memref<7x14x21xf32>) {
  linalg.reciprocal ins(%arg : memref<7x14x21xf32>) outs(%out : memref<7x14x21xf32>)
  return
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK: func @generalize_reciprocal
// CHECK-SAME: (%[[ARG:.+]]: memref<7x14x21xf32>, %[[OUT:.+]]: memref<7x14x21xf32>)

// CHECK: %[[one:.+]] = arith.constant 1.000000e+00 : f32

// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]}
// CHECK-SAME:  ins(%[[LHS]] : memref<7x14x21xf32>) outs(%[[OUT]] : memref<7x14x21xf32>)

// CHECK:         ^{{.+}}(%[[BBARG0:.+]]: f32, %[[BBARG1:.+]]: f32)
// CHECK-NEXT:      %[[reciprocal:.+]] = arith.divf %[[one]], %[[BBARG0]] : f32
// CHECK-NEXT:      linalg.yield %[[reciprocal]] : f32

// -----

func.func @generalize_round(%arg: memref<7x14x21xf32>, %out: memref<7x14x21xf32>) {
  linalg.round ins(%arg : memref<7x14x21xf32>) outs(%out : memref<7x14x21xf32>)
  return
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK: func @generalize_round
// CHECK-SAME: (%[[ARG:.+]]: memref<7x14x21xf32>, %[[OUT:.+]]: memref<7x14x21xf32>)

// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]}
// CHECK-SAME:  ins(%[[LHS]] : memref<7x14x21xf32>) outs(%[[OUT]] : memref<7x14x21xf32>)

// CHECK:         ^{{.+}}(%[[BBARG0:.+]]: f32, %[[BBARG1:.+]]: f32)
// CHECK-NEXT:      %[[round:.+]] = math.round %[[BBARG0]] : f32
// CHECK-NEXT:      linalg.yield %[[round]] : f32

// -----

func.func @generalize_sqrt(%arg: memref<7x14x21xf32>, %out: memref<7x14x21xf32>) {
  linalg.sqrt ins(%arg : memref<7x14x21xf32>) outs(%out : memref<7x14x21xf32>)
  return
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK: func @generalize_sqrt
// CHECK-SAME: (%[[ARG:.+]]: memref<7x14x21xf32>, %[[OUT:.+]]: memref<7x14x21xf32>)

// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]}
// CHECK-SAME:  ins(%[[LHS]] : memref<7x14x21xf32>) outs(%[[OUT]] : memref<7x14x21xf32>)

// CHECK:         ^{{.+}}(%[[BBARG0:.+]]: f32, %[[BBARG1:.+]]: f32)
// CHECK-NEXT:      %[[sqrt:.+]] = math.sqrt %[[BBARG0]] : f32
// CHECK-NEXT:      linalg.yield %[[sqrt]] : f32

// -----

func.func @generalize_rsqrt(%arg: memref<7x14x21xf32>, %out: memref<7x14x21xf32>) {
  linalg.rsqrt ins(%arg : memref<7x14x21xf32>) outs(%out : memref<7x14x21xf32>)
  return
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK: func @generalize_rsqrt
// CHECK-SAME: (%[[ARG:.+]]: memref<7x14x21xf32>, %[[OUT:.+]]: memref<7x14x21xf32>)

// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]}
// CHECK-SAME:  ins(%[[LHS]] : memref<7x14x21xf32>) outs(%[[OUT]] : memref<7x14x21xf32>)

// CHECK:         ^{{.+}}(%[[BBARG0:.+]]: f32, %[[BBARG1:.+]]: f32)
// CHECK-NEXT:      %[[rsqrt:.+]] = math.rsqrt %[[BBARG0]] : f32
// CHECK-NEXT:      linalg.yield %[[rsqrt]] : f32

// -----

func.func @generalize_square(%arg: memref<7x14x21xf32>, %out: memref<7x14x21xf32>) {
  linalg.square ins(%arg : memref<7x14x21xf32>) outs(%out : memref<7x14x21xf32>)
  return
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK: func @generalize_square
// CHECK-SAME: (%[[ARG:.+]]: memref<7x14x21xf32>, %[[OUT:.+]]: memref<7x14x21xf32>)

// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]}
// CHECK-SAME:  ins(%[[LHS]] : memref<7x14x21xf32>) outs(%[[OUT]] : memref<7x14x21xf32>)

// CHECK:         ^{{.+}}(%[[BBARG0:.+]]: f32, %[[BBARG1:.+]]: f32)
// CHECK-NEXT:      %[[square:.+]] = arith.mulf %[[BBARG0]], %[[BBARG0]] : f32
// CHECK-NEXT:      linalg.yield %[[square]] : f32

// -----

func.func @generalize_tanh(%arg: memref<7x14x21xf32>, %out: memref<7x14x21xf32>) {
  linalg.tanh ins(%arg : memref<7x14x21xf32>) outs(%out : memref<7x14x21xf32>)
  return
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK: func @generalize_tanh
// CHECK-SAME: (%[[ARG:.+]]: memref<7x14x21xf32>, %[[OUT:.+]]: memref<7x14x21xf32>)

// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]}
// CHECK-SAME:  ins(%[[LHS]] : memref<7x14x21xf32>) outs(%[[OUT]] : memref<7x14x21xf32>)

// CHECK:         ^{{.+}}(%[[BBARG0:.+]]: f32, %[[BBARG1:.+]]: f32)
// CHECK-NEXT:      %[[tanh:.+]] = math.tanh %[[BBARG0]] : f32
// CHECK-NEXT:      linalg.yield %[[tanh]] : f32

// -----

func.func @generalize_erf(%arg: memref<7x14x21xf32>, %out: memref<7x14x21xf32>) {
  linalg.erf ins(%arg : memref<7x14x21xf32>) outs(%out : memref<7x14x21xf32>)
  return
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK: func @generalize_erf
// CHECK-SAME: (%[[ARG:.+]]: memref<7x14x21xf32>, %[[OUT:.+]]: memref<7x14x21xf32>)

// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]}
// CHECK-SAME:  ins(%[[LHS]] : memref<7x14x21xf32>) outs(%[[OUT]] : memref<7x14x21xf32>)

// CHECK:         ^{{.+}}(%[[BBARG0:.+]]: f32, %[[BBARG1:.+]]: f32)
// CHECK-NEXT:      %[[erf:.+]] = math.erf %[[BBARG0]] : f32
// CHECK-NEXT:      linalg.yield %[[erf]] : f32

// -----

func.func @generalize_max(%lhs: memref<7x14x21xf32>, %rhs: memref<7x14x21xf32>,
                          %out: memref<7x14x21xf32>) {
  linalg.max ins(%lhs, %rhs : memref<7x14x21xf32>, memref<7x14x21xf32>)
             outs(%out : memref<7x14x21xf32>)
  return
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK: func @generalize_max
// CHECK-SAME: (%[[LHS:.+]]: memref<7x14x21xf32>, %[[RHS:.+]]: memref<7x14x21xf32>,
// CHECK-SAME:  %[[OUT:.+]]: memref<7x14x21xf32>)

// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP]], #[[MAP]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]}
// CHECK-SAME:  ins(%[[LHS]], %[[RHS]] : memref<7x14x21xf32>, memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)

// CHECK:         ^{{.+}}(%[[BBARG0:.+]]: f32, %[[BBARG1:.+]]: f32, %[[BBARG2:.+]]: f32)
// CHECK-NEXT:      %[[max:.+]] = arith.maximumf %[[BBARG0]], %[[BBARG1]] : f32
// CHECK-NEXT:      linalg.yield %[[max]] : f32

// -----

func.func @generalize_min(%lhs: memref<7x14x21xf32>, %rhs: memref<7x14x21xf32>,
                          %out: memref<7x14x21xf32>) {
  linalg.min ins(%lhs, %rhs : memref<7x14x21xf32>, memref<7x14x21xf32>)
             outs(%out : memref<7x14x21xf32>)
  return
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK: func @generalize_min
// CHECK-SAME: (%[[LHS:.+]]: memref<7x14x21xf32>, %[[RHS:.+]]: memref<7x14x21xf32>,
// CHECK-SAME:  %[[OUT:.+]]: memref<7x14x21xf32>)

// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP]], #[[MAP]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]}
// CHECK-SAME:  ins(%[[LHS]], %[[RHS]] : memref<7x14x21xf32>, memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)

// CHECK:         ^{{.+}}(%[[BBARG0:.+]]: f32, %[[BBARG1:.+]]: f32, %[[BBARG2:.+]]: f32)
// CHECK-NEXT:      %[[min:.+]] = arith.minimumf %[[BBARG0]], %[[BBARG1]] : f32
// CHECK-NEXT:      linalg.yield %[[min]] : f32


// -----

func.func @generalize_powf(%lhs: memref<7x14x21xf32>, %rhs: memref<7x14x21xf32>,
                          %out: memref<7x14x21xf32>) {
  linalg.powf ins(%lhs, %rhs : memref<7x14x21xf32>, memref<7x14x21xf32>)
             outs(%out : memref<7x14x21xf32>)
  return
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK: func @generalize_powf
// CHECK-SAME: (%[[LHS:.+]]: memref<7x14x21xf32>, %[[RHS:.+]]: memref<7x14x21xf32>,
// CHECK-SAME:  %[[OUT:.+]]: memref<7x14x21xf32>)

// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP]], #[[MAP]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]}
// CHECK-SAME:  ins(%[[LHS]], %[[RHS]] : memref<7x14x21xf32>, memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)

// CHECK:         ^{{.+}}(%[[BBARG0:.+]]: f32, %[[BBARG1:.+]]: f32, %[[BBARG2:.+]]: f32)
// CHECK-NEXT:      %[[powf:.+]] = math.powf %[[BBARG0]], %[[BBARG1]] : f32
// CHECK-NEXT:      linalg.yield %[[powf]] : f32


// -----

func.func @generalize_select(%cond: memref<7x14x21xi1>, %lhs: memref<7x14x21xf32>, %rhs: memref<7x14x21xf32>,
                              %out: memref<7x14x21xf32>) {
  linalg.select ins(%cond, %lhs, %rhs: memref<7x14x21xi1>, memref<7x14x21xf32>, memref<7x14x21xf32>)
                outs(%out: memref<7x14x21xf32>)
  return
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK: func @generalize_select
// CHECK-SAME: (%[[COND:.+]]: memref<7x14x21xi1>, %[[LHS:.+]]: memref<7x14x21xf32>, %[[RHS:.+]]: memref<7x14x21xf32>,
// CHECK-SAME:  %[[OUT:.+]]: memref<7x14x21xf32>)

// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP]], #[[MAP]], #[[MAP]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]}
// CHECK-SAME:  ins(%[[COND]], %[[LHS]], %[[RHS]] : memref<7x14x21xi1>, memref<7x14x21xf32>, memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)

// CHECK:         ^{{.+}}(%[[BBARG0:.+]]: i1, %[[BBARG1:.+]]: f32, %[[BBARG2:.+]]: f32, %[[BBARG3:.+]]: f32)
// CHECK-NEXT:      %[[select:.+]] = arith.select %[[BBARG0]], %[[BBARG1]], %[[BBARG2]] : f32
// CHECK-NEXT:      linalg.yield %[[select]] : f32


// -----

// CHECK-LABEL: func @fill_tensor
func.func @fill_tensor(%f: f32, %v: vector<2x4xf32>) -> (tensor<f32>, tensor<vector<2x4xf32>>) {
  %e0 = tensor.empty() : tensor<f32>
  %0 = linalg.fill ins(%f : f32) outs(%e0 : tensor<f32>) -> tensor<f32>
// CHECK: linalg.generic
// CHECK:         ^{{.+}}(%[[BBARG0:.+]]: f32, %[[BBARG1:.+]]: f32)
// CHECK-NEXT:      linalg.yield %[[BBARG0]] : f32

  %e1 = tensor.empty() : tensor<vector<2x4xf32>>
  %1 = linalg.fill ins(%v : vector<2x4xf32>) outs(%e1 : tensor<vector<2x4xf32>>) -> tensor<vector<2x4xf32>>
// CHECK: linalg.generic
// CHECK:         ^{{.+}}(%[[BBARG0:.+]]: vector<2x4xf32>, %[[BBARG1:.+]]: vector<2x4xf32>)
// CHECK-NEXT:      linalg.yield %[[BBARG0]] : vector<2x4xf32>

  return %0, %1: tensor<f32>, tensor<vector<2x4xf32>>
}

// -----

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d2, d0)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL:   func.func @matmul_transpose_a_explicit(
// CHECK-SAME:                                  %[[VAL_0:.*]]: memref<5x3xf32>,
// CHECK-SAME:                                  %[[VAL_1:.*]]: memref<5x7xf32>,
// CHECK-SAME:                                  %[[VAL_2:.*]]: memref<3x7xf32>) {

// CHECK:           linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel", "reduction"]}
// CHECK:           arith.mulf
// CHECK:           arith.addf

func.func @matmul_transpose_a_explicit(%arg0: memref<5x3xf32>, %arg1: memref<5x7xf32>, %arg2: memref<3x7xf32>) {
  linalg.matmul indexing_maps = [
                       affine_map<(d0, d1, d2) -> (d2, d0)>,
                       affine_map<(d0, d1, d2) -> (d2, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>
                      ]
                      ins(%arg0, %arg1 : memref<5x3xf32>, memref<5x7xf32>)
                      outs(%arg2: memref<3x7xf32>)
                      
  return
}

// -----

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL:   func.func @matmul_transpose_b_explicit(
// CHECK-SAME:                                           %[[VAL_0:.*]]: memref<3x5xf32>,
// CHECK-SAME:                                           %[[VAL_1:.*]]: memref<7x5xf32>,
// CHECK-SAME:                                           %[[VAL_2:.*]]: memref<3x7xf32>) {

// CHECK:           linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel", "reduction"]}
// CHECK:           arith.mulf
// CHECK:           arith.addf

func.func @matmul_transpose_b_explicit(%arg0: memref<3x5xf32>, %arg1: memref<7x5xf32>, %arg2: memref<3x7xf32>) {
  linalg.matmul indexing_maps = [
                       affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>
                      ]
                      ins(%arg0, %arg1 : memref<3x5xf32>, memref<7x5xf32>)
                      outs(%arg2: memref<3x7xf32>)
                      
  return
}

// -----

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d2, d0)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL:   func.func @matmul_transpose_a_b_explicit(
// CHECK-SAME:                                             %[[VAL_0:.*]]: memref<5x3xf32>,
// CHECK-SAME:                                             %[[VAL_1:.*]]: memref<7x5xf32>,
// CHECK-SAME:                                             %[[VAL_2:.*]]: memref<3x7xf32>) {

// CHECK:           linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel", "reduction"]}
// CHECK:           arith.mulf
// CHECK:           arith.addf

func.func @matmul_transpose_a_b_explicit(%arg0: memref<5x3xf32>, %arg1: memref<7x5xf32>, %arg2: memref<3x7xf32>) {
  linalg.matmul indexing_maps = [
                       affine_map<(d0, d1, d2) -> (d2, d0)>,
                       affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>
                      ]
                      ins(%arg0, %arg1 : memref<5x3xf32>, memref<7x5xf32>)
                      outs(%arg2: memref<3x7xf32>)
                      
  return
}

// -----

