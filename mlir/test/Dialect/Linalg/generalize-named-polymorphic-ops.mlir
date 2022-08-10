// RUN: mlir-opt %s -split-input-file -linalg-generalize-named-ops | FileCheck %s

// Verifies that different argument types is legal.
func.func @generalize_matmul_tensor_f16f64f32(%A : tensor<16x8xf16>, %B: tensor<8x32xf64>, %C: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = linalg.matmul ins(%A, %B: tensor<16x8xf16>, tensor<8x32xf64>)
                          outs(%C: tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0: tensor<16x32xf32>
}

// CHECK-LABEL: @generalize_matmul_tensor_f16f64f32
// CHECK:      ^{{.*}}(%[[A_ARG:.+]]: f16, %[[B_ARG:.+]]: f64, %[[C_ARG:.+]]: f32)
// Verify floating point extension and truncation.
// CHECK-NEXT:   %[[A_CAST:.+]] = arith.extf %[[A_ARG]] : f16 to f32
// CHECK-NEXT:   %[[B_CAST:.+]] = arith.truncf %[[B_ARG]] : f64 to f32
// CHECK-NEXT:   %[[MUL:.+]] = arith.mulf %[[A_CAST]], %[[B_CAST]] : f32
// CHECK-NEXT:   %[[ADD:.+]] = arith.addf %[[C_ARG]], %[[MUL]] : f32
// CHECK-NEXT:   linalg.yield %[[ADD]] : f32
// CHECK-NEXT: -> tensor<16x32xf32>

// -----

// Verifies that different argument types is legal.
func.func @generalize_matmul_tensor_i16i64i32(%A : tensor<16x8xi16>, %B: tensor<8x32xi64>, %C: tensor<16x32xi32>) -> tensor<16x32xi32> {
  %0 = linalg.matmul ins(%A, %B: tensor<16x8xi16>, tensor<8x32xi64>)
                          outs(%C: tensor<16x32xi32>) -> tensor<16x32xi32>
  return %0: tensor<16x32xi32>
}

// CHECK-LABEL: @generalize_matmul_tensor_i16i64i32
// CHECK:      ^{{.*}}(%[[A_ARG:.+]]: i16, %[[B_ARG:.+]]: i64, %[[C_ARG:.+]]: i32)
// Verify signed integer extension and truncation.
// CHECK-NEXT:   %[[A_CAST:.+]] = arith.extsi %[[A_ARG]] : i16 to i32
// CHECK-NEXT:   %[[B_CAST:.+]] = arith.trunci %[[B_ARG]] : i64 to i32
// CHECK-NEXT:   %[[MUL:.+]] = arith.muli %[[A_CAST]], %[[B_CAST]] : i32
// CHECK-NEXT:   %[[ADD:.+]] = arith.addi %[[C_ARG]], %[[MUL]] : i32
// CHECK-NEXT:   linalg.yield %[[ADD]] : i32
// CHECK-NEXT: -> tensor<16x32xi32>


// -----

// Verifies that cast attributes control the cast operations used.
func.func @generalize_matmul_tensor_i16i64i32_unsigned(%A : tensor<16x8xi16>, %B: tensor<8x32xi64>, %C: tensor<16x32xi32>) -> tensor<16x32xi32> {
  %0 = linalg.matmul {cast = #linalg.type_fn<cast_unsigned>}
                     ins(%A, %B: tensor<16x8xi16>, tensor<8x32xi64>)
                          outs(%C: tensor<16x32xi32>) -> tensor<16x32xi32>
  return %0: tensor<16x32xi32>
}

// CHECK-LABEL: @generalize_matmul_tensor_i16i64i32_unsigned
// CHECK:        = arith.extui

// -----

func.func @generalize_matmul_tensor_i16i64f32(%A : tensor<16x8xi16>, %B: tensor<8x32xi64>, %C: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = linalg.matmul ins(%A, %B: tensor<16x8xi16>, tensor<8x32xi64>)
                     outs(%C: tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0: tensor<16x32xf32>
}

// CHECK-LABEL: @generalize_matmul_tensor_i16i64f32
// Verify signed integer to floating point cast.
// CHECK:        = arith.sitofp
// CHECK:        = arith.sitofp

// -----

func.func @generalize_matmul_tensor_f16f64i32(%A : tensor<16x8xf16>, %B: tensor<8x32xf64>, %C: tensor<16x32xi32>) -> tensor<16x32xi32> {
  %0 = linalg.matmul ins(%A, %B: tensor<16x8xf16>, tensor<8x32xf64>)
                              outs(%C: tensor<16x32xi32>) -> tensor<16x32xi32>
  return %0: tensor<16x32xi32>
}

// CHECK-LABEL: @generalize_matmul_tensor_f16f64i32
// Verify floating point to signed integer cast.
// CHECK:        = arith.fptosi
// CHECK:        = arith.fptosi

// -----

func.func @generalize_matmul_unsigned_tensor_i16i64i32(%A : tensor<16x8xi16>, %B: tensor<8x32xi64>, %C: tensor<16x32xi32>) -> tensor<16x32xi32> {
  %0 = linalg.matmul_unsigned ins(%A, %B: tensor<16x8xi16>, tensor<8x32xi64>)
                              outs(%C: tensor<16x32xi32>) -> tensor<16x32xi32>
  return %0: tensor<16x32xi32>
}

// CHECK-LABEL: @generalize_matmul_unsigned_tensor_i16i64i32
// Verify unsigned integer extension and truncation.
// CHECK:        = arith.extui
// CHECK:        = arith.trunci

// -----

func.func @generalize_matmul_unsigned_tensor_i16i64f32(%A : tensor<16x8xi16>, %B: tensor<8x32xi64>, %C: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = linalg.matmul_unsigned ins(%A, %B: tensor<16x8xi16>, tensor<8x32xi64>)
                              outs(%C: tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0: tensor<16x32xf32>
}

// CHECK-LABEL: @generalize_matmul_unsigned_tensor_i16i64f32
// Verify unsigned integer to floating point cast.
// CHECK:        = arith.uitofp
// CHECK:        = arith.uitofp

// -----

func.func @generalize_matmul_unsigned_tensor_f16f64i32(%A : tensor<16x8xf16>, %B: tensor<8x32xf64>, %C: tensor<16x32xi32>) -> tensor<16x32xi32> {
  %0 = linalg.matmul_unsigned ins(%A, %B: tensor<16x8xf16>, tensor<8x32xf64>)
                              outs(%C: tensor<16x32xi32>) -> tensor<16x32xi32>
  return %0: tensor<16x32xi32>
}

// CHECK-LABEL: @generalize_matmul_unsigned_tensor_f16f64i32
// Verify floating point to unsigend integer cast.
// CHECK:        = arith.fptoui
// CHECK:        = arith.fptoui

// -----

func.func @generalize_pooling_nhwc_max_f32(%input : tensor<1x4x16x1xf32>, %shape: tensor<2x2xf32>, %output: tensor<1x2x4x1xf32>) -> tensor<1x2x4x1xf32> {
  %0 = linalg.pooling_nhwc_max {dilations = dense<[1, 2]> : tensor<2xi64>, strides = dense<[2, 4]> : tensor<2xi64>}
    ins(%input, %shape : tensor<1x4x16x1xf32>, tensor<2x2xf32>) outs(%output : tensor<1x2x4x1xf32>) -> tensor<1x2x4x1xf32>
  return %0: tensor<1x2x4x1xf32>
}

// CHECK-LABEL: @generalize_pooling_nhwc_max_f32
// CHECK:      ^{{.*}}(%[[IN_ARG:.+]]: f32, %[[SHAPE_ARG:.+]]: f32, %[[OUT_ARG:.+]]: f32)
// CHECK-NEXT:   %[[MAX:.+]] = arith.maxf %[[OUT_ARG]], %[[IN_ARG]] : f32
// CHECK-NEXT:   linalg.yield %[[MAX]] : f32
// CHECK-NEXT: -> tensor<1x2x4x1xf32>

// -----

func.func @generalize_pooling_nhwc_max_i32(%input : tensor<1x4x16x1xi32>, %shape: tensor<2x2xi32>, %output: tensor<1x2x4x1xi32>) -> tensor<1x2x4x1xi32> {
  %0 = linalg.pooling_nhwc_max {dilations = dense<[1, 2]> : tensor<2xi64>, strides = dense<[2, 4]> : tensor<2xi64>}
    ins(%input, %shape : tensor<1x4x16x1xi32>, tensor<2x2xi32>) outs(%output : tensor<1x2x4x1xi32>) -> tensor<1x2x4x1xi32>
  return %0: tensor<1x2x4x1xi32>
}

// CHECK-LABEL: @generalize_pooling_nhwc_max_i32
// Verify signed integer maximum.
// CHECK:        = arith.maxsi

// -----

func.func @generalize_pooling_nhwc_max_unsigned_i32(%input : tensor<1x4x16x1xi32>, %shape: tensor<2x2xi32>, %output: tensor<1x2x4x1xi32>) -> tensor<1x2x4x1xi32> {
  %0 = linalg.pooling_nhwc_max_unsigned {dilations = dense<[1, 2]> : tensor<2xi64>, strides = dense<[2, 4]> : tensor<2xi64>}
    ins(%input, %shape : tensor<1x4x16x1xi32>, tensor<2x2xi32>) outs(%output : tensor<1x2x4x1xi32>) -> tensor<1x2x4x1xi32>
  return %0: tensor<1x2x4x1xi32>
}

// CHECK-LABEL: @generalize_pooling_nhwc_max_unsigned_i32
// Verify unsigned integer minimum.
// CHECK:        = arith.maxui

// -----

func.func @generalize_pooling_nhwc_min_f32(%input : tensor<1x4x16x1xf32>, %shape: tensor<2x2xf32>, %output: tensor<1x2x4x1xf32>) -> tensor<1x2x4x1xf32> {
  %0 = linalg.pooling_nhwc_min {dilations = dense<[1, 2]> : tensor<2xi64>, strides = dense<[2, 4]> : tensor<2xi64>}
    ins(%input, %shape : tensor<1x4x16x1xf32>, tensor<2x2xf32>) outs(%output : tensor<1x2x4x1xf32>) -> tensor<1x2x4x1xf32>
  return %0: tensor<1x2x4x1xf32>
}

// CHECK-LABEL: @generalize_pooling_nhwc_min_f32
// CHECK:      ^{{.*}}(%[[IN_ARG:.+]]: f32, %[[SHAPE_ARG:.+]]: f32, %[[OUT_ARG:.+]]: f32)
// CHECK-NEXT:   %[[MIN:.+]] = arith.minf %[[OUT_ARG]], %[[IN_ARG]] : f32
// CHECK-NEXT:   linalg.yield %[[MIN]] : f32
// CHECK-NEXT: -> tensor<1x2x4x1xf32>

// -----

func.func @generalize_pooling_nhwc_min_i32(%input : tensor<1x4x16x1xi32>, %shape: tensor<2x2xi32>, %output: tensor<1x2x4x1xi32>) -> tensor<1x2x4x1xi32> {
  %0 = linalg.pooling_nhwc_min {dilations = dense<[1, 2]> : tensor<2xi64>, strides = dense<[2, 4]> : tensor<2xi64>}
    ins(%input, %shape : tensor<1x4x16x1xi32>, tensor<2x2xi32>) outs(%output : tensor<1x2x4x1xi32>) -> tensor<1x2x4x1xi32>
  return %0: tensor<1x2x4x1xi32>
}

// CHECK-LABEL: @generalize_pooling_nhwc_min_i32
// Verify signed integer minimum.
// CHECK:        = arith.minsi

// -----

func.func @generalize_pooling_nhwc_min_unsigned_i32(%input : tensor<1x4x16x1xi32>, %shape: tensor<2x2xi32>, %output: tensor<1x2x4x1xi32>) -> tensor<1x2x4x1xi32> {
  %0 = linalg.pooling_nhwc_min_unsigned {dilations = dense<[1, 2]> : tensor<2xi64>, strides = dense<[2, 4]> : tensor<2xi64>}
    ins(%input, %shape : tensor<1x4x16x1xi32>, tensor<2x2xi32>) outs(%output : tensor<1x2x4x1xi32>) -> tensor<1x2x4x1xi32>
  return %0: tensor<1x2x4x1xi32>
}

// CHECK-LABEL: @generalize_pooling_nhwc_min_unsigned_i32
// Verify unsigned integer minimum.
// CHECK:        = arith.minui

// -----

func.func @generalize_pooling_nhwc_sum_f32(%input : tensor<1x4x16x1xf32>, %shape: tensor<2x2xf32>, %output: tensor<1x2x4x1xf32>) -> tensor<1x2x4x1xf32> {
  %0 = linalg.pooling_nhwc_sum {dilations = dense<[1, 2]> : tensor<2xi64>, strides = dense<[2, 4]> : tensor<2xi64>}
    ins(%input, %shape : tensor<1x4x16x1xf32>, tensor<2x2xf32>) outs(%output : tensor<1x2x4x1xf32>) -> tensor<1x2x4x1xf32>
  return %0: tensor<1x2x4x1xf32>
}

// CHECK-LABEL: @generalize_pooling_nhwc_sum_f32
// CHECK:      ^{{.*}}(%[[IN_ARG:.+]]: f32, %[[SHAPE_ARG:.+]]: f32, %[[OUT_ARG:.+]]: f32)
// CHECK-NEXT:   %[[ADD:.+]] = arith.addf %[[OUT_ARG]], %[[IN_ARG]] : f32
// CHECK-NEXT:   linalg.yield %[[ADD]] : f32
// CHECK-NEXT: -> tensor<1x2x4x1xf32>

// -----

func.func @generalize_pooling_nhwc_sum_i32(%input : tensor<1x4x16x1xi32>, %shape: tensor<2x2xi32>, %output: tensor<1x2x4x1xi32>) -> tensor<1x2x4x1xi32> {
  %0 = linalg.pooling_nhwc_sum {dilations = dense<[1, 2]> : tensor<2xi64>, strides = dense<[2, 4]> : tensor<2xi64>}
    ins(%input, %shape : tensor<1x4x16x1xi32>, tensor<2x2xi32>) outs(%output : tensor<1x2x4x1xi32>) -> tensor<1x2x4x1xi32>
  return %0: tensor<1x2x4x1xi32>
}

// CHECK-LABEL: @generalize_pooling_nhwc_sum_i32
// CHECK:      ^{{.*}}(%[[IN_ARG:.+]]: i32, %[[SHAPE_ARG:.+]]: i32, %[[OUT_ARG:.+]]: i32)
// CHECK-NEXT:   %[[ADD:.+]] = arith.addi %[[OUT_ARG]], %[[IN_ARG]] : i32
// CHECK-NEXT:   linalg.yield %[[ADD]] : i32
// CHECK-NEXT: -> tensor<1x2x4x1xi32>

// -----

func.func @generalize_fill_0d(%value: f64, %O: tensor<f32>) -> tensor<f32> {
  %0 = linalg.fill ins(%value: f64) outs(%O : tensor<f32>) -> tensor<f32>
  return %0: tensor<f32>
}

// CHECK-DAG: #[[$MAP0:.+]] = affine_map<() -> ()>

// CHECK-LABEL: @generalize_fill_0d
// CHECK:      linalg.generic
// CHECK-SAME: indexing_maps = [#[[$MAP0]], #[[$MAP0]]]
// CHECK-SAME: iterator_types = []

// -----

func.func @generalize_fill_2d(%value: f64, %O: memref<16x32xf32>) {
  linalg.fill ins(%value: f64) outs(%O : memref<16x32xf32>)
  return
}

// CHECK-DAG: #[[$MAP0:.+]] = affine_map<(d0, d1) -> ()>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @generalize_fill
// CHECK:      linalg.generic
// CHECK-SAME: indexing_maps = [#[[$MAP0]], #[[$MAP1]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel"]

// -----

func.func @generalize_index(%min: f64, %max: f64, %seed: i32, %O: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = linalg.fill_rng_2d ins(%min, %max, %seed: f64, f64, i32) outs(%O : tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0: tensor<16x32xf32>
}

// CHECK-LABEL: @generalize_index
// CHECK-DAG:    %[[IDX0:.+]] = linalg.index 0 : index
// CHECK-DAG:    %[[IDX1:.+]] = linalg.index 1 : index
// CHECK-DAG:    %[[IDX0_CAST:.+]] = arith.index_cast %[[IDX0]] : index to i32
// CHECK-DAG:    %[[IDX1_CAST:.+]] = arith.index_cast %[[IDX1]] : index to i32

// -----

func.func @generalize_const(%min: f64, %max: f64, %seed: i32, %O: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = linalg.fill_rng_2d ins(%min, %max, %seed: f64, f64, i32) outs(%O : tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0: tensor<16x32xf32>
}

// CHECK-LABEL: @generalize_const
// CHECK-DAG:    %[[CST0:.+]] = arith.constant 1103515245 : i32
// CHECK-DAG:    %[[CST1:.+]] = arith.constant 12345 : i32
// CHECK-DAG:    %[[CST2:.+]] = arith.constant 2.3283063999999999E-10 : f64

// -----

// Verifies the default value of the fun attribute is an exp op.
func.func @generalize_elemwise_exp(%lhs : tensor<4x8xf32>, %output : tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = linalg.elemwise_unary ins(%lhs: tensor<4x8xf32>) outs(%output: tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0: tensor<4x8xf32>
}

// CHECK-LABEL: @generalize_elemwise_exp
// CHECK:        = math.exp

// -----

// Verifies the fun attribute controls the unary function used.
func.func @generalize_elemwise_log(%lhs : tensor<4x8xf32>, %output : tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>}
                              ins(%lhs: tensor<4x8xf32>) outs(%output: tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0: tensor<4x8xf32>
}

// CHECK-LABEL: @generalize_elemwise_log
// CHECK:        = math.log

// -----

// Verifies the fun attribute controls the unary function used.
func.func @generalize_elemwise_abs(%lhs : tensor<4x8xf32>, %output : tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>}
                              ins(%lhs: tensor<4x8xf32>) outs(%output: tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0: tensor<4x8xf32>
}

// CHECK-LABEL: @generalize_elemwise_abs
// CHECK:        = math.absf

// -----

// Verifies the fun attribute controls the unary function used.
func.func @generalize_elemwise_ceil(%lhs : tensor<4x8xf32>, %output : tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = linalg.elemwise_unary {fun = #linalg.unary_fn<ceil>}
                              ins(%lhs: tensor<4x8xf32>) outs(%output: tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0: tensor<4x8xf32>
}

// CHECK-LABEL: @generalize_elemwise_ceil
// CHECK:        = math.ceil

// -----

// Verifies the fun attribute controls the unary function used.
func.func @generalize_elemwise_floor(%lhs : tensor<4x8xf32>, %output : tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = linalg.elemwise_unary {fun = #linalg.unary_fn<floor>}
                              ins(%lhs: tensor<4x8xf32>) outs(%output: tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0: tensor<4x8xf32>
}

// CHECK-LABEL: @generalize_elemwise_floor
// CHECK:        = math.floor

// -----

// Verifies the fun attribute controls the unary function used.
func.func @generalize_elemwise_negf(%lhs : tensor<4x8xf32>, %output : tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>}
                              ins(%lhs: tensor<4x8xf32>) outs(%output: tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0: tensor<4x8xf32>
}

// CHECK-LABEL: @generalize_elemwise_negf
// CHECK:        = arith.negf

// -----

// Verifies the default value of the fun attribute is an add op.
func.func @generalize_elemwise_add(%lhs : tensor<4x8xf32>, %rhs : tensor<4x8xf32>, %output : tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = linalg.elemwise_binary ins(%lhs, %rhs: tensor<4x8xf32>, tensor<4x8xf32>)
                              outs(%output: tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0: tensor<4x8xf32>
}

// CHECK-LABEL: @generalize_elemwise_add
// CHECK:        = arith.addf

// -----

// Verifies the fun attribute controls the binary function used.
func.func @generalize_elemwise_mul(%lhs : tensor<4x8xf32>, %rhs : tensor<4x8xf32>, %output : tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
                              ins(%lhs, %rhs: tensor<4x8xf32>, tensor<4x8xf32>)
                              outs(%output: tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0: tensor<4x8xf32>
}

// CHECK-LABEL: @generalize_elemwise_mul
// CHECK:        = arith.mulf

// -----

// Verifies pointwise ops support rank zero input tensors
func.func @generalize_elemwise_rank_zero(%lhs : tensor<f32>, %rhs : tensor<f32>, %output : tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}
                              ins(%lhs, %rhs: tensor<f32>, tensor<f32>)
                              outs(%output: tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0: tensor<4x8xf32>
}

// CHECK-LABEL: @generalize_elemwise_rank_zero
// CHECK:       linalg.generic
// CHECK-SAME:  iterator_types = ["parallel", "parallel"]
// CHECK:        = arith.subf

// -----

// Verifies the fun attribute controls the binary function used.
func.func @generalize_copy(%lhs : tensor<4x8xf32>, %output : tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = linalg.copy ins(%lhs: tensor<4x8xf32>) outs(%output: tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0: tensor<4x8xf32>
}

// CHECK-LABEL: @generalize_copy
//       CHECK:   linalg.generic
//  CHECK-NEXT:   ^bb0(%[[I:[0-9a-zA-Z]*]]: f32
//  CHECK-NEXT:   linalg.yield %[[I]]
