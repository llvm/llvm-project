// RUN: mlir-opt %s -split-input-file -linalg-generalize-named-ops | FileCheck %s

func @generalize_matmul_tensor_f32(%A : tensor<16x8xf32>, %B: tensor<8x32xf32>, %C: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = linalg.polymorphic_matmul ins(%A, %B: tensor<16x8xf32>, tensor<8x32xf32>)
                          outs(%C: tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0: tensor<16x32xf32>
}

// CHECK:      ^{{.*}}(%[[A_ARG:.+]]: f32, %[[B_ARG:.+]]: f32, %[[C_ARG:.+]]: f32)
// CHECK-NEXT:   %[[MUL:.+]] = mulf %[[A_ARG]], %[[B_ARG]] : f32
// CHECK-NEXT:   %[[ADD:.+]] = addf %[[C_ARG]], %[[MUL]] : f32
// CHECK-NEXT:   linalg.yield %[[ADD]] : f32
// CHECK-NEXT: -> tensor<16x32xf32>

// -----

func @generalize_matmul_tensor_i32(%A : tensor<16x8xi32>, %B: tensor<8x32xi32>, %C: tensor<16x32xi32>) -> tensor<16x32xi32> {
  %0 = linalg.polymorphic_matmul ins(%A, %B: tensor<16x8xi32>, tensor<8x32xi32>)
                          outs(%C: tensor<16x32xi32>) -> tensor<16x32xi32>
  return %0: tensor<16x32xi32>
}

// CHECK:      ^{{.*}}(%[[A_ARG:.+]]: i32, %[[B_ARG:.+]]: i32, %[[C_ARG:.+]]: i32)
// CHECK-NEXT:   %[[MUL:.+]] = muli %[[A_ARG]], %[[B_ARG]] : i32
// CHECK-NEXT:   %[[ADD:.+]] = addi %[[C_ARG]], %[[MUL]] : i32
// CHECK-NEXT:   linalg.yield %[[ADD]] : i32
// CHECK-NEXT: -> tensor<16x32xi32>

// -----
// Verifies floating point to integer cast.
func @generalize_matmul_tensor_f32_f32_i16(%A : tensor<16x8xf32>, %B: tensor<8x32xf32>, %C: tensor<16x32xi16>) -> tensor<16x32xi16> {
  %0 = linalg.polymorphic_matmul ins(%A, %B: tensor<16x8xf32>, tensor<8x32xf32>)
                          outs(%C: tensor<16x32xi16>) -> tensor<16x32xi16>
  return %0: tensor<16x32xi16>
}

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
  %0 = linalg.polymorphic_matmul ins(%A, %B: tensor<16x8xi8>, tensor<8x32xi8>)
                          outs(%C: tensor<16x32xi32>) -> tensor<16x32xi32>
  return %0: tensor<16x32xi32>
}

// -----
// Verifies that different argument types is legal.
func @generalize_matmul_tensor_i8_i16_i32(%A : tensor<16x8xi8>, %B: tensor<8x32xi16>, %C: tensor<16x32xi32>) -> tensor<16x32xi32> {
  %0 = linalg.polymorphic_matmul ins(%A, %B: tensor<16x8xi8>, tensor<8x32xi16>)
                          outs(%C: tensor<16x32xi32>) -> tensor<16x32xi32>
  return %0: tensor<16x32xi32>
}

// CHECK:      ^{{.*}}(%[[A_ARG:.+]]: i8, %[[B_ARG:.+]]: i8, %[[C_ARG:.+]]: i32)
// CHECK-NEXT:   %[[A_CAST:.+]] = sexti %[[A_ARG]] : i8 to i32
// CHECK-NEXT:   %[[B_CAST:.+]] = sexti %[[B_ARG]] : i8 to i32
// CHECK-NEXT:   %[[MUL:.+]] = muli %[[A_CAST]], %[[B_CAST]] : i32
// CHECK-NEXT:   %[[ADD:.+]] = addi %[[C_ARG]], %[[MUL]] : i32
// CHECK-NEXT:   linalg.yield %[[ADD]] : i32
// CHECK-NEXT: -> tensor<16x32xi32>

// -----
// Somewhat non-sensical but checks integer truncation cast.
func @generalize_matmul_tensor_i32_i32_i16(%A : tensor<16x8xi32>, %B: tensor<8x32xi32>, %C: tensor<16x32xi16>) -> tensor<16x32xi16> {
  %0 = linalg.polymorphic_matmul ins(%A, %B: tensor<16x8xi32>, tensor<8x32xi32>)
                          outs(%C: tensor<16x32xi16>) -> tensor<16x32xi16>
  return %0: tensor<16x32xi16>
}

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
  %0 = linalg.polymorphic_matmul ins(%A, %B: tensor<16x8xi8>, tensor<8x32xi8>)
                          outs(%C: tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0: tensor<16x32xf32>
}

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
  %0 = linalg.polymorphic_matmul ins(%A, %B: tensor<16x8xf16>, tensor<8x32xf16>)
                          outs(%C: tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0: tensor<16x32xf32>
}

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
  %0 = linalg.polymorphic_matmul ins(%A, %B: tensor<16x8xf64>, tensor<8x32xf64>)
                          outs(%C: tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0: tensor<16x32xf32>
}

// CHECK:      ^{{.*}}(%[[A_ARG:.+]]: f64, %[[B_ARG:.+]]: f64, %[[C_ARG:.+]]: f32)
// CHECK-NEXT:   %[[A_CAST:.+]] = fptrunc %[[A_ARG]] : f64 to f32
// CHECK-NEXT:   %[[B_CAST:.+]] = fptrunc %[[B_ARG]] : f64 to f32
// CHECK-NEXT:   %[[MUL:.+]] = mulf %[[A_CAST]], %[[B_CAST]] : f32
// CHECK-NEXT:   %[[ADD:.+]] = addf %[[C_ARG]], %[[MUL]] : f32
// CHECK-NEXT:   linalg.yield %[[ADD]] : f32
// CHECK-NEXT: -> tensor<16x32xf32>
