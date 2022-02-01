// RUN: mlir-opt %s -split-input-file -linalg-generalize-named-ops | FileCheck %s

// Verifies that different argument types is legal.
func @generalize_matmul_tensor_f16f64f32(%A : tensor<16x8xf16>, %B: tensor<8x32xf64>, %C: tensor<16x32xf32>) -> tensor<16x32xf32> {
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
func @generalize_matmul_tensor_i16i64i32(%A : tensor<16x8xi16>, %B: tensor<8x32xi64>, %C: tensor<16x32xi32>) -> tensor<16x32xi32> {
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

func @generalize_matmul_tensor_i16i64f32(%A : tensor<16x8xi16>, %B: tensor<8x32xi64>, %C: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = linalg.matmul ins(%A, %B: tensor<16x8xi16>, tensor<8x32xi64>)
                     outs(%C: tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0: tensor<16x32xf32>
}

// CHECK-LABEL: @generalize_matmul_tensor_i16i64f32
// Verify signed integer to floating point cast.
// CHECK:        = arith.sitofp
// CHECK:        = arith.sitofp

// -----

func @generalize_matmul_tensor_f16f64i32(%A : tensor<16x8xf16>, %B: tensor<8x32xf64>, %C: tensor<16x32xi32>) -> tensor<16x32xi32> {
  %0 = linalg.matmul ins(%A, %B: tensor<16x8xf16>, tensor<8x32xf64>)
                              outs(%C: tensor<16x32xi32>) -> tensor<16x32xi32>
  return %0: tensor<16x32xi32>
}

// CHECK-LABEL: @generalize_matmul_tensor_f16f64i32
// Verify floating point to signed integer cast.
// CHECK:        = arith.fptosi
// CHECK:        = arith.fptosi

// -----

func @generalize_matmul_unsigned_tensor_i16i64i32(%A : tensor<16x8xi16>, %B: tensor<8x32xi64>, %C: tensor<16x32xi32>) -> tensor<16x32xi32> {
  %0 = linalg.matmul_unsigned ins(%A, %B: tensor<16x8xi16>, tensor<8x32xi64>)
                              outs(%C: tensor<16x32xi32>) -> tensor<16x32xi32>
  return %0: tensor<16x32xi32>
}

// CHECK-LABEL: @generalize_matmul_unsigned_tensor_i16i64i32
// Verify unsigned integer extension and truncation.
// CHECK:        = arith.extui
// CHECK:        = arith.trunci

// -----

func @generalize_matmul_unsigned_tensor_i16i64f32(%A : tensor<16x8xi16>, %B: tensor<8x32xi64>, %C: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = linalg.matmul_unsigned ins(%A, %B: tensor<16x8xi16>, tensor<8x32xi64>)
                              outs(%C: tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0: tensor<16x32xf32>
}

// CHECK-LABEL: @generalize_matmul_unsigned_tensor_i16i64f32
// Verify unsigned integer to floating point cast.
// CHECK:        = arith.uitofp
// CHECK:        = arith.uitofp

// -----

func @generalize_matmul_unsigned_tensor_f16f64i32(%A : tensor<16x8xf16>, %B: tensor<8x32xf64>, %C: tensor<16x32xi32>) -> tensor<16x32xi32> {
  %0 = linalg.matmul_unsigned ins(%A, %B: tensor<16x8xf16>, tensor<8x32xf64>)
                              outs(%C: tensor<16x32xi32>) -> tensor<16x32xi32>
  return %0: tensor<16x32xi32>
}

// CHECK-LABEL: @generalize_matmul_unsigned_tensor_f16f64i32
// Verify floating point to unsigend integer cast.
// CHECK:        = arith.fptoui
// CHECK:        = arith.fptoui

// -----

func @generalize_pooling_nhwc_max_f32(%input : tensor<1x4x16x1xf32>, %shape: tensor<2x2xf32>, %output: tensor<1x2x4x1xf32>) -> tensor<1x2x4x1xf32> {
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

func @generalize_pooling_nhwc_max_i32(%input : tensor<1x4x16x1xi32>, %shape: tensor<2x2xi32>, %output: tensor<1x2x4x1xi32>) -> tensor<1x2x4x1xi32> {
  %0 = linalg.pooling_nhwc_max {dilations = dense<[1, 2]> : tensor<2xi64>, strides = dense<[2, 4]> : tensor<2xi64>}
    ins(%input, %shape : tensor<1x4x16x1xi32>, tensor<2x2xi32>) outs(%output : tensor<1x2x4x1xi32>) -> tensor<1x2x4x1xi32>
  return %0: tensor<1x2x4x1xi32>
}

// CHECK-LABEL: @generalize_pooling_nhwc_max_i32
// Verify signed integer maximum.
// CHECK:        = arith.maxsi

// -----

func @generalize_pooling_nhwc_max_unsigned_i32(%input : tensor<1x4x16x1xi32>, %shape: tensor<2x2xi32>, %output: tensor<1x2x4x1xi32>) -> tensor<1x2x4x1xi32> {
  %0 = linalg.pooling_nhwc_max_unsigned {dilations = dense<[1, 2]> : tensor<2xi64>, strides = dense<[2, 4]> : tensor<2xi64>}
    ins(%input, %shape : tensor<1x4x16x1xi32>, tensor<2x2xi32>) outs(%output : tensor<1x2x4x1xi32>) -> tensor<1x2x4x1xi32>
  return %0: tensor<1x2x4x1xi32>
}

// CHECK-LABEL: @generalize_pooling_nhwc_max_unsigned_i32
// Verify unsigned integer minimum.
// CHECK:        = arith.maxui

// -----

func @generalize_pooling_nhwc_min_f32(%input : tensor<1x4x16x1xf32>, %shape: tensor<2x2xf32>, %output: tensor<1x2x4x1xf32>) -> tensor<1x2x4x1xf32> {
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

func @generalize_pooling_nhwc_min_i32(%input : tensor<1x4x16x1xi32>, %shape: tensor<2x2xi32>, %output: tensor<1x2x4x1xi32>) -> tensor<1x2x4x1xi32> {
  %0 = linalg.pooling_nhwc_min {dilations = dense<[1, 2]> : tensor<2xi64>, strides = dense<[2, 4]> : tensor<2xi64>}
    ins(%input, %shape : tensor<1x4x16x1xi32>, tensor<2x2xi32>) outs(%output : tensor<1x2x4x1xi32>) -> tensor<1x2x4x1xi32>
  return %0: tensor<1x2x4x1xi32>
}

// CHECK-LABEL: @generalize_pooling_nhwc_min_i32
// Verify signed integer minimum.
// CHECK:        = arith.minsi

// -----

func @generalize_pooling_nhwc_min_unsigned_i32(%input : tensor<1x4x16x1xi32>, %shape: tensor<2x2xi32>, %output: tensor<1x2x4x1xi32>) -> tensor<1x2x4x1xi32> {
  %0 = linalg.pooling_nhwc_min_unsigned {dilations = dense<[1, 2]> : tensor<2xi64>, strides = dense<[2, 4]> : tensor<2xi64>}
    ins(%input, %shape : tensor<1x4x16x1xi32>, tensor<2x2xi32>) outs(%output : tensor<1x2x4x1xi32>) -> tensor<1x2x4x1xi32>
  return %0: tensor<1x2x4x1xi32>
}

// CHECK-LABEL: @generalize_pooling_nhwc_min_unsigned_i32
// Verify unsigned integer minimum.
// CHECK:        = arith.minui

// -----

func @generalize_pooling_nhwc_sum_f32(%input : tensor<1x4x16x1xf32>, %shape: tensor<2x2xf32>, %output: tensor<1x2x4x1xf32>) -> tensor<1x2x4x1xf32> {
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

func @generalize_pooling_nhwc_sum_i32(%input : tensor<1x4x16x1xi32>, %shape: tensor<2x2xi32>, %output: tensor<1x2x4x1xi32>) -> tensor<1x2x4x1xi32> {
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

func @generalize_fill_rng_2d_f32(%min: f64, %max: f64, %seed: i32, %O: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = linalg.fill_rng_2d ins(%min, %max, %seed: f64, f64, i32) outs(%O : tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0: tensor<16x32xf32>
}

// CHECK-LABEL: @generalize_fill_rng_2d_f32
// CHECK-DAG:  ^{{.*}}(%[[MIN:.+]]: f64, %[[MAX:.+]]: f64, %[[SEED:.+]]: i32, %[[O:.+]]: f32
// CHECK-DAG:    %[[IDX0:.+]] = linalg.index 0 : index
// CHECK-DAG:    %[[IDX1:.+]] = linalg.index 1 : index
// CHECK-DAG:    %[[IDX0_CAST:.+]] = arith.index_cast %[[IDX0]] : index to i32
// CHECK-DAG:    %[[IDX1_CAST:.+]] = arith.index_cast %[[IDX1]] : index to i32
// CHECK-DAG:    %[[VAL0:.+]] = arith.addi %[[IDX0_CAST]], %[[SEED]] : i32
// CHECK-DAG:    %[[CST0:.+]] = arith.constant 1103515245 : i32
// CHECK-DAG:    %[[CST1:.+]] = arith.constant 12345 : i32
// CHECK-DAG:    %[[VAL1:.+]] = arith.muli %[[VAL0]], %[[CST0]] : i32
// CHECK-DAG:    %[[VAL2:.+]] = arith.addi %[[VAL1]], %[[CST1]] : i32
// Skip random number computation for the second index.
// CHECK-DAG:    %[[DIFF:.+]] = arith.subf %[[MAX]], %[[MIN]] : f64
// CHECK-DAG:    %[[CST2:.+]] = arith.constant 2.3283063999999999E-10 : f64
// CHECK-DAG:    %[[FACT:.+]] = arith.mulf %[[DIFF]], %[[CST2]] : f64
// CHECK-DAG:    %[[VAL4:.+]] = arith.mulf %{{.+}}, %[[FACT]] : f64
// CHECK-DAG:    %[[VAL5:.+]] = arith.addf %[[VAL4]], %[[MIN]] : f64
// CHECK-DAG:    %[[VAL6:.+]] = arith.truncf %[[VAL5]] : f64 to f32
// CHECK-NEXT:   linalg.yield %[[VAL6]] : f32
// CHECK-NEXT: -> tensor<16x32xf32>

// -----

func @generalize_fill_rng_2d_i32(%min: f64, %max: f64, %seed: i32, %O: tensor<16x32xi32>) -> tensor<16x32xi32> {
  %0 = linalg.fill_rng_2d ins(%min, %max, %seed: f64, f64, i32) outs(%O : tensor<16x32xi32>) -> tensor<16x32xi32>
  return %0: tensor<16x32xi32>
}

// CHECK-LABEL: @generalize_fill_rng_2d_i32
// CHECK: ^{{.*}}(%[[MIN:.+]]: f64, %[[MAX:.+]]: f64, %[[SEED:.+]]: i32, %[[O:.+]]: i32
// Verifies floating point to integer cast.
// CHECK:        %[[VAL6:.+]] = arith.fptosi %{{.+}} : f64 to i32
// CHECK-NEXT:   linalg.yield %[[VAL6]] : i32
// CHECK-NEXT: -> tensor<16x32xi32>

// -----

func @generalize_soft_plus_2d_f32(%input: tensor<16x32xf32>, %output: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = linalg.soft_plus_2d ins(%input: tensor<16x32xf32>) outs(%output: tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0: tensor<16x32xf32>
}

// CHECK-LABEL: @generalize_soft_plus_2d_f32
//      CHECK: %[[C1:.+]] = arith.constant 1.000000e+00 : f32
//      CHECK: ^{{.*}}(%[[IN:.+]]: f32, %[[OUT:.+]]: f32
// CHECK-NEXT:   %[[EXP:.+]] = math.exp %[[IN]] : f32
// CHECK-NEXT:   %[[SUM:.+]] = arith.addf %[[EXP]], %[[C1]] : f32
// CHECK-NEXT:   %[[LOG:.+]] = math.log %[[SUM]] : f32
// CHECK-NEXT:   linalg.yield %[[LOG]] : f32
// CHECK-NEXT: -> tensor<16x32xf32>
