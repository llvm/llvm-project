// RUN: mlir-opt %s -split-input-file -linalg-generalize-named-ops | FileCheck %s

func @generalize_matmul_tensor_f32(%A : tensor<16x8xf32>, %B: tensor<8x32xf32>, %C: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = linalg.matmul ins(%A, %B: tensor<16x8xf32>, tensor<8x32xf32>)
                          outs(%C: tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0: tensor<16x32xf32>
}

// CHECK-LABEL: @generalize_matmul_tensor_f32
// CHECK:      ^{{.*}}(%[[A_ARG:.+]]: f32, %[[B_ARG:.+]]: f32, %[[C_ARG:.+]]: f32)
// CHECK-NEXT:   %[[MUL:.+]] = mulf %[[A_ARG]], %[[B_ARG]] : f32
// CHECK-NEXT:   %[[ADD:.+]] = addf %[[C_ARG]], %[[MUL]] : f32
// CHECK-NEXT:   linalg.yield %[[ADD]] : f32
// CHECK-NEXT: -> tensor<16x32xf32>

// -----

func @generalize_matmul_tensor_i32(%A : tensor<16x8xi32>, %B: tensor<8x32xi32>, %C: tensor<16x32xi32>) -> tensor<16x32xi32> {
  %0 = linalg.matmul ins(%A, %B: tensor<16x8xi32>, tensor<8x32xi32>)
                          outs(%C: tensor<16x32xi32>) -> tensor<16x32xi32>
  return %0: tensor<16x32xi32>
}

// CHECK-LABEL: @generalize_matmul_tensor_i32
// CHECK:      ^{{.*}}(%[[A_ARG:.+]]: i32, %[[B_ARG:.+]]: i32, %[[C_ARG:.+]]: i32)
// CHECK-NEXT:   %[[MUL:.+]] = muli %[[A_ARG]], %[[B_ARG]] : i32
// CHECK-NEXT:   %[[ADD:.+]] = addi %[[C_ARG]], %[[MUL]] : i32
// CHECK-NEXT:   linalg.yield %[[ADD]] : i32
// CHECK-NEXT: -> tensor<16x32xi32>

// -----

func @generalize_depthwise_conv_2d_input_nhwc_filter_hwc_poly_f32(%input : tensor<1x4x16x1xf32>, %filter: tensor<2x2x1xf32>, %output: tensor<1x2x4x1xf32>) -> tensor<1x2x4x1xf32> {
  %0 = linalg.depthwise_conv_2d_input_nhwc_filter_hwc_poly {dilations = dense<[1, 2]> : tensor<2xi64>, strides = dense<[2, 4]> : tensor<2xi64>}
    ins(%input, %filter : tensor<1x4x16x1xf32>, tensor<2x2x1xf32>) outs(%output : tensor<1x2x4x1xf32>) -> tensor<1x2x4x1xf32>
  return %0: tensor<1x2x4x1xf32>
}

// CHECK-LABEL: @generalize_depthwise_conv_2d_input_nhwc_filter_hwc_poly_f32
// CHECK:      ^{{.*}}(%[[IN_ARG:.+]]: f32, %[[FILTER_ARG:.+]]: f32, %[[OUT_ARG:.+]]: f32)
// CHECK-NEXT:   %[[MUL:.+]] = mulf %[[IN_ARG]], %[[FILTER_ARG]] : f32
// CHECK-NEXT:   %[[ADD:.+]] = addf %[[OUT_ARG]], %[[MUL]] : f32
// CHECK-NEXT:   linalg.yield %[[ADD]] : f32
// CHECK-NEXT: -> tensor<1x2x4x1xf32>

// -----

func @generalize_depthwise_conv_2d_input_nhwc_filter_hwc_poly_i32(%input : tensor<1x4x16x1xi32>, %filter: tensor<2x2x1xi32>, %output: tensor<1x2x4x1xi32>) -> tensor<1x2x4x1xi32> {
  %0 = linalg.depthwise_conv_2d_input_nhwc_filter_hwc_poly {dilations = dense<[1, 2]> : tensor<2xi64>, strides = dense<[2, 4]> : tensor<2xi64>}
    ins(%input, %filter : tensor<1x4x16x1xi32>, tensor<2x2x1xi32>) outs(%output : tensor<1x2x4x1xi32>) -> tensor<1x2x4x1xi32>
  return %0: tensor<1x2x4x1xi32>
}

// CHECK-LABEL: @generalize_depthwise_conv_2d_input_nhwc_filter_hwc_poly_i32
// CHECK:      ^{{.*}}(%[[IN_ARG:.+]]: i32, %[[FILTER_ARG:.+]]: i32, %[[OUT_ARG:.+]]: i32)
// CHECK-NEXT:   %[[MUL:.+]] = muli %[[IN_ARG]], %[[FILTER_ARG]] : i32
// CHECK-NEXT:   %[[ADD:.+]] = addi %[[OUT_ARG]], %[[MUL]] : i32
// CHECK-NEXT:   linalg.yield %[[ADD]] : i32
// CHECK-NEXT: -> tensor<1x2x4x1xi32>

// -----

func @generalize_pooling_nhwc_max_f32(%input : tensor<1x4x16x1xf32>, %shape: tensor<2x2xf32>, %output: tensor<1x2x4x1xf32>) -> tensor<1x2x4x1xf32> {
  %0 = linalg.pooling_nhwc_max {dilations = dense<[1, 2]> : tensor<2xi64>, strides = dense<[2, 4]> : tensor<2xi64>}
    ins(%input, %shape : tensor<1x4x16x1xf32>, tensor<2x2xf32>) outs(%output : tensor<1x2x4x1xf32>) -> tensor<1x2x4x1xf32>
  return %0: tensor<1x2x4x1xf32>
}

// CHECK-LABEL: @generalize_pooling_nhwc_max_f32
// CHECK:      ^{{.*}}(%[[IN_ARG:.+]]: f32, %[[SHAPE_ARG:.+]]: f32, %[[OUT_ARG:.+]]: f32)
// CHECK-NEXT:   %[[COND:.+]] = cmpf ogt, %[[OUT_ARG]], %[[IN_ARG]] : f32
// CHECK-NEXT:   %[[MAX:.+]] = select %[[COND]], %[[OUT_ARG]], %[[IN_ARG]] : f32
// CHECK-NEXT:   linalg.yield %[[MAX]] : f32
// CHECK-NEXT: -> tensor<1x2x4x1xf32>

// -----

func @generalize_pooling_nhwc_max_i32(%input : tensor<1x4x16x1xi32>, %shape: tensor<2x2xi32>, %output: tensor<1x2x4x1xi32>) -> tensor<1x2x4x1xi32> {
  %0 = linalg.pooling_nhwc_max {dilations = dense<[1, 2]> : tensor<2xi64>, strides = dense<[2, 4]> : tensor<2xi64>}
    ins(%input, %shape : tensor<1x4x16x1xi32>, tensor<2x2xi32>) outs(%output : tensor<1x2x4x1xi32>) -> tensor<1x2x4x1xi32>
  return %0: tensor<1x2x4x1xi32>
}

// CHECK-LABEL: @generalize_pooling_nhwc_max_i32
// CHECK:      ^{{.*}}(%[[IN_ARG:.+]]: i32, %[[SHAPE_ARG:.+]]: i32, %[[OUT_ARG:.+]]: i32)
// CHECK-NEXT:   %[[COND:.+]] = cmpi sgt, %[[OUT_ARG]], %[[IN_ARG]] : i32
// CHECK-NEXT:   %[[MAX:.+]] = select %[[COND]], %[[OUT_ARG]], %[[IN_ARG]] : i32
// CHECK-NEXT:   linalg.yield %[[MAX]] : i32
// CHECK-NEXT: -> tensor<1x2x4x1xi32>

// -----

func @generalize_pooling_nhwc_min_f32(%input : tensor<1x4x16x1xf32>, %shape: tensor<2x2xf32>, %output: tensor<1x2x4x1xf32>) -> tensor<1x2x4x1xf32> {
  %0 = linalg.pooling_nhwc_min {dilations = dense<[1, 2]> : tensor<2xi64>, strides = dense<[2, 4]> : tensor<2xi64>}
    ins(%input, %shape : tensor<1x4x16x1xf32>, tensor<2x2xf32>) outs(%output : tensor<1x2x4x1xf32>) -> tensor<1x2x4x1xf32>
  return %0: tensor<1x2x4x1xf32>
}

// CHECK-LABEL: @generalize_pooling_nhwc_min_f32
// CHECK:      ^{{.*}}(%[[IN_ARG:.+]]: f32, %[[SHAPE_ARG:.+]]: f32, %[[OUT_ARG:.+]]: f32)
// CHECK-NEXT:   %[[COND:.+]] = cmpf olt, %[[OUT_ARG]], %[[IN_ARG]] : f32
// CHECK-NEXT:   %[[MAX:.+]] = select %[[COND]], %[[OUT_ARG]], %[[IN_ARG]] : f32
// CHECK-NEXT:   linalg.yield %[[MAX]] : f32
// CHECK-NEXT: -> tensor<1x2x4x1xf32>

// -----

func @generalize_pooling_nhwc_min_i32(%input : tensor<1x4x16x1xi32>, %shape: tensor<2x2xi32>, %output: tensor<1x2x4x1xi32>) -> tensor<1x2x4x1xi32> {
  %0 = linalg.pooling_nhwc_min {dilations = dense<[1, 2]> : tensor<2xi64>, strides = dense<[2, 4]> : tensor<2xi64>}
    ins(%input, %shape : tensor<1x4x16x1xi32>, tensor<2x2xi32>) outs(%output : tensor<1x2x4x1xi32>) -> tensor<1x2x4x1xi32>
  return %0: tensor<1x2x4x1xi32>
}

// CHECK-LABEL: @generalize_pooling_nhwc_min_i32
// CHECK:      ^{{.*}}(%[[IN_ARG:.+]]: i32, %[[SHAPE_ARG:.+]]: i32, %[[OUT_ARG:.+]]: i32)
// CHECK-NEXT:   %[[COND:.+]] = cmpi slt, %[[OUT_ARG]], %[[IN_ARG]] : i32
// CHECK-NEXT:   %[[MAX:.+]] = select %[[COND]], %[[OUT_ARG]], %[[IN_ARG]] : i32
// CHECK-NEXT:   linalg.yield %[[MAX]] : i32
// CHECK-NEXT: -> tensor<1x2x4x1xi32>

// -----

func @generalize_pooling_nhwc_sum_f32(%input : tensor<1x4x16x1xf32>, %shape: tensor<2x2xf32>, %output: tensor<1x2x4x1xf32>) -> tensor<1x2x4x1xf32> {
  %0 = linalg.pooling_nhwc_sum {dilations = dense<[1, 2]> : tensor<2xi64>, strides = dense<[2, 4]> : tensor<2xi64>}
    ins(%input, %shape : tensor<1x4x16x1xf32>, tensor<2x2xf32>) outs(%output : tensor<1x2x4x1xf32>) -> tensor<1x2x4x1xf32>
  return %0: tensor<1x2x4x1xf32>
}

// CHECK-LABEL: @generalize_pooling_nhwc_sum_f32
// CHECK:      ^{{.*}}(%[[IN_ARG:.+]]: f32, %[[SHAPE_ARG:.+]]: f32, %[[OUT_ARG:.+]]: f32)
// CHECK-NEXT:   %[[ADD:.+]] = addf %[[OUT_ARG]], %[[IN_ARG]] : f32
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
// CHECK-NEXT:   %[[ADD:.+]] = addi %[[OUT_ARG]], %[[IN_ARG]] : i32
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
// CHECK-DAG:    %[[IDX0_CAST:.+]] = index_cast %[[IDX0]] : index to i32
// CHECK-DAG:    %[[IDX1_CAST:.+]] = index_cast %[[IDX1]] : index to i32
// CHECK-DAG:    %[[VAL0:.+]] = addi %[[IDX0_CAST]], %[[SEED]] : i32
// CHECK-DAG:    %[[CST0:.+]] = constant 1103515245 : i32
// CHECK-DAG:    %[[CST1:.+]] = constant 12345 : i32
// CHECK-DAG:    %[[VAL1:.+]] = muli %[[VAL0]], %[[CST0]] : i32
// CHECK-DAG:    %[[VAL2:.+]] = addi %[[VAL1]], %[[CST1]] : i32
// Skip random number computation for the second index.
// CHECK-DAG:    %[[DIFF:.+]] = subf %[[MAX]], %[[MIN]] : f64
// CHECK-DAG:    %[[CST2:.+]] = constant 2.3283063999999999E-10 : f64
// CHECK-DAG:    %[[FACT:.+]] = mulf %[[DIFF]], %[[CST2]] : f64
// CHECK-DAG:    %[[VAL4:.+]] = mulf %{{.+}}, %[[FACT]] : f64
// CHECK-DAG:    %[[VAL5:.+]] = addf %[[VAL4]], %[[MIN]] : f64
// CHECK-DAG:    %[[VAL6:.+]] = fptrunc %[[VAL5]] : f64 to f32
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
// CHECK:        %[[VAL6:.+]] = fptosi %{{.+}} : f64 to i32
// CHECK-NEXT:   linalg.yield %[[VAL6]] : i32
// CHECK-NEXT: -> tensor<16x32xi32>

// -----

func @generalize_soft_plus_2d_f32(%input: tensor<16x32xf32>, %output: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = linalg.soft_plus_2d ins(%input: tensor<16x32xf32>) outs(%output: tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0: tensor<16x32xf32>
}

// CHECK-LABEL: @generalize_soft_plus_2d_f32
//      CHECK: %[[C1:.+]] = constant 1.000000e+00 : f32
//      CHECK: ^{{.*}}(%[[IN:.+]]: f32, %[[OUT:.+]]: f32
// CHECK-NEXT:   %[[EXP:.+]] = math.exp %[[IN]] : f32
// CHECK-NEXT:   %[[SUM:.+]] = addf %[[C1]], %[[EXP]] : f32
// CHECK-NEXT:   %[[LOG:.+]] = math.log %[[SUM]] : f32
// CHECK-NEXT:   linalg.yield %[[LOG]] : f32
// CHECK-NEXT: -> tensor<16x32xf32>

// -----
// Verifies floating point to integer cast.
func @generalize_matmul_tensor_f32_f32_i16(%A : tensor<16x8xf32>, %B: tensor<8x32xf32>, %C: tensor<16x32xi16>) -> tensor<16x32xi16> {
  %0 = linalg.matmul ins(%A, %B: tensor<16x8xf32>, tensor<8x32xf32>)
                          outs(%C: tensor<16x32xi16>) -> tensor<16x32xi16>
  return %0: tensor<16x32xi16>
}

// CHECK-LABEL: @generalize_matmul_tensor_f32_f32_i16
// CHECK:      ^{{.*}}(%[[A_ARG:.+]]: f32, %[[B_ARG:.+]]: f32, %[[C_ARG:.+]]: i16)
// CHECK-NEXT:   %[[A_CAST:.+]] = fptosi %[[A_ARG]] : f32 to i16
// CHECK-NEXT:   %[[B_CAST:.+]] = fptosi %[[B_ARG]] : f32 to i16
// CHECK-NEXT:   %[[MUL:.+]] = muli %[[A_CAST]], %[[B_CAST]] : i16
// CHECK-NEXT:   %[[ADD:.+]] = addi %[[C_ARG]], %[[MUL]] : i16
// CHECK-NEXT:   linalg.yield %[[ADD]] : i16
// CHECK-NEXT: -> tensor<16x32xi16>

// -----
// Verifies sign extension cast.
func @generalize_matmul_tensor_i8_i8_i32(%A : tensor<16x8xi8>, %B: tensor<8x32xi8>, %C: tensor<16x32xi32>) -> tensor<16x32xi32> {
  %0 = linalg.matmul ins(%A, %B: tensor<16x8xi8>, tensor<8x32xi8>)
                          outs(%C: tensor<16x32xi32>) -> tensor<16x32xi32>
  return %0: tensor<16x32xi32>
}

// CHECK-LABEL: @generalize_matmul_tensor_i8_i8_i32
// CHECK:      ^{{.*}}(%[[A_ARG:.+]]: i8, %[[B_ARG:.+]]: i8, %[[C_ARG:.+]]: i32)
// CHECK-NEXT:   %[[A_CAST:.+]] = sexti %[[A_ARG]] : i8 to i32
// CHECK-NEXT:   %[[B_CAST:.+]] = sexti %[[B_ARG]] : i8 to i32
// CHECK-NEXT:   %[[MUL:.+]] = muli %[[A_CAST]], %[[B_CAST]] : i32
// CHECK-NEXT:   %[[ADD:.+]] = addi %[[C_ARG]], %[[MUL]] : i32
// CHECK-NEXT:   linalg.yield %[[ADD]] : i32
// CHECK-NEXT: -> tensor<16x32xi32>

// -----
// Verifies that different argument types is legal.
func @generalize_matmul_tensor_i8_i16_i32(%A : tensor<16x8xi8>, %B: tensor<8x32xi16>, %C: tensor<16x32xi32>) -> tensor<16x32xi32> {
  %0 = linalg.matmul ins(%A, %B: tensor<16x8xi8>, tensor<8x32xi16>)
                          outs(%C: tensor<16x32xi32>) -> tensor<16x32xi32>
  return %0: tensor<16x32xi32>
}

// CHECK-LABEL: @generalize_matmul_tensor_i8_i16_i32
// CHECK:      ^{{.*}}(%[[A_ARG:.+]]: i8, %[[B_ARG:.+]]: i16, %[[C_ARG:.+]]: i32)
// CHECK-NEXT:   %[[A_CAST:.+]] = sexti %[[A_ARG]] : i8 to i32
// CHECK-NEXT:   %[[B_CAST:.+]] = sexti %[[B_ARG]] : i16 to i32
// CHECK-NEXT:   %[[MUL:.+]] = muli %[[A_CAST]], %[[B_CAST]] : i32
// CHECK-NEXT:   %[[ADD:.+]] = addi %[[C_ARG]], %[[MUL]] : i32
// CHECK-NEXT:   linalg.yield %[[ADD]] : i32
// CHECK-NEXT: -> tensor<16x32xi32>

// -----
// Somewhat non-sensical but checks integer truncation cast.
func @generalize_matmul_tensor_i32_i32_i16(%A : tensor<16x8xi32>, %B: tensor<8x32xi32>, %C: tensor<16x32xi16>) -> tensor<16x32xi16> {
  %0 = linalg.matmul ins(%A, %B: tensor<16x8xi32>, tensor<8x32xi32>)
                          outs(%C: tensor<16x32xi16>) -> tensor<16x32xi16>
  return %0: tensor<16x32xi16>
}

// CHECK-LABEL: @generalize_matmul_tensor_i32_i32_i16
// CHECK:      ^{{.*}}(%[[A_ARG:.+]]: i32, %[[B_ARG:.+]]: i32, %[[C_ARG:.+]]: i16)
// CHECK-NEXT:   %[[A_CAST:.+]] = trunci %[[A_ARG]] : i32 to i16
// CHECK-NEXT:   %[[B_CAST:.+]] = trunci %[[B_ARG]] : i32 to i16
// CHECK-NEXT:   %[[MUL:.+]] = muli %[[A_CAST]], %[[B_CAST]] : i16
// CHECK-NEXT:   %[[ADD:.+]] = addi %[[C_ARG]], %[[MUL]] : i16
// CHECK-NEXT:   linalg.yield %[[ADD]] : i16
// CHECK-NEXT: -> tensor<16x32xi16>

// -----
// Verifies integer to floating point cast.
func @generalize_matmul_tensor_i8_i8_f32(%A : tensor<16x8xi8>, %B: tensor<8x32xi8>, %C: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = linalg.matmul ins(%A, %B: tensor<16x8xi8>, tensor<8x32xi8>)
                          outs(%C: tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0: tensor<16x32xf32>
}

// CHECK-LABEL: @generalize_matmul_tensor_i8_i8_f32
// CHECK:      ^{{.*}}(%[[A_ARG:.+]]: i8, %[[B_ARG:.+]]: i8, %[[C_ARG:.+]]: f32)
// CHECK-NEXT:   %[[A_CAST:.+]] = sitofp %[[A_ARG]] : i8 to f32
// CHECK-NEXT:   %[[B_CAST:.+]] = sitofp %[[B_ARG]] : i8 to f32
// CHECK-NEXT:   %[[MUL:.+]] = mulf %[[A_CAST]], %[[B_CAST]] : f32
// CHECK-NEXT:   %[[ADD:.+]] = addf %[[C_ARG]], %[[MUL]] : f32
// CHECK-NEXT:   linalg.yield %[[ADD]] : f32
// CHECK-NEXT: -> tensor<16x32xf32>

// -----
// Verifies floating point extension cast.
func @generalize_matmul_tensor_f16_f16_f32(%A : tensor<16x8xf16>, %B: tensor<8x32xf16>, %C: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = linalg.matmul ins(%A, %B: tensor<16x8xf16>, tensor<8x32xf16>)
                          outs(%C: tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0: tensor<16x32xf32>
}

// CHECK-LABEL: @generalize_matmul_tensor_f16_f16_f32
// CHECK:      ^{{.*}}(%[[A_ARG:.+]]: f16, %[[B_ARG:.+]]: f16, %[[C_ARG:.+]]: f32)
// CHECK-NEXT:   %[[A_CAST:.+]] = fpext %[[A_ARG]] : f16 to f32
// CHECK-NEXT:   %[[B_CAST:.+]] = fpext %[[B_ARG]] : f16 to f32
// CHECK-NEXT:   %[[MUL:.+]] = mulf %[[A_CAST]], %[[B_CAST]] : f32
// CHECK-NEXT:   %[[ADD:.+]] = addf %[[C_ARG]], %[[MUL]] : f32
// CHECK-NEXT:   linalg.yield %[[ADD]] : f32
// CHECK-NEXT: -> tensor<16x32xf32>

// -----
// Verifies floating point truncation.
func @generalize_matmul_tensor_f64_f64_f32(%A : tensor<16x8xf64>, %B: tensor<8x32xf64>, %C: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = linalg.matmul ins(%A, %B: tensor<16x8xf64>, tensor<8x32xf64>)
                          outs(%C: tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0: tensor<16x32xf32>
}

// CHECK-LABEL: @generalize_matmul_tensor_f64_f64_f32
// CHECK:      ^{{.*}}(%[[A_ARG:.+]]: f64, %[[B_ARG:.+]]: f64, %[[C_ARG:.+]]: f32)
// CHECK-NEXT:   %[[A_CAST:.+]] = fptrunc %[[A_ARG]] : f64 to f32
// CHECK-NEXT:   %[[B_CAST:.+]] = fptrunc %[[B_ARG]] : f64 to f32
// CHECK-NEXT:   %[[MUL:.+]] = mulf %[[A_CAST]], %[[B_CAST]] : f32
// CHECK-NEXT:   %[[ADD:.+]] = addf %[[C_ARG]], %[[MUL]] : f32
// CHECK-NEXT:   linalg.yield %[[ADD]] : f32
// CHECK-NEXT: -> tensor<16x32xf32>
