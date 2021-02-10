// RUN: mlir-opt %s -test-linalg-transform-patterns=test-linalg-to-vector-patterns -split-input-file | FileCheck %s

// -----

// CHECK-LABEL: contraction_dot
func @contraction_dot(%A: memref<1584xf32>, %B: memref<1584xf32>, %C: memref<f32>) {
  // CHECK: vector.contract
  // CHECK-SAME: vector<1584xf32>, vector<1584xf32> into f32
  linalg.dot ins(%A, %B: memref<1584xf32>, memref<1584xf32>)
            outs(%C: memref<f32>)
  return
}

// -----

// CHECK-LABEL: contraction_matvec
func @contraction_matvec(%A: memref<1584x1584xf32>, %B: memref<1584xf32>, %C: memref<1584xf32>) {
  // CHECK: vector.contract
  // CHECK-SAME: vector<1584x1584xf32>, vector<1584xf32> into vector<1584xf32>
  linalg.matvec ins(%A, %B: memref<1584x1584xf32>, memref<1584xf32>)
            outs(%C: memref<1584xf32>)
  return
}

// -----

// CHECK-LABEL: contraction_matmul
func @contraction_matmul(%A: memref<1584x1584xf32>, %B: memref<1584x1584xf32>, %C: memref<1584x1584xf32>) {
  // CHECK: vector.contract
  // CHECK-SAME: vector<1584x1584xf32>, vector<1584x1584xf32> into vector<1584x1584xf32>
  linalg.matmul ins(%A, %B: memref<1584x1584xf32>, memref<1584x1584xf32>)
            outs(%C: memref<1584x1584xf32>)
  return
}

// -----

// CHECK-LABEL: contraction_batch_matmul
func @contraction_batch_matmul(%A: memref<1584x1584x1584xf32>, %B: memref<1584x1584x1584xf32>, %C: memref<1584x1584x1584xf32>) {
  // CHECK: vector.contract
  // CHECK-SAME: vector<1584x1584x1584xf32>, vector<1584x1584x1584xf32> into vector<1584x1584x1584xf32>
  linalg.batch_matmul
    ins(%A, %B: memref<1584x1584x1584xf32>, memref<1584x1584x1584xf32>)
   outs(%C: memref<1584x1584x1584xf32>)
  return
}

// -----

#matmul_trait = {
  args_in = 2,
  args_out = 1,
  indexing_maps = [
    affine_map<(m, n, k) -> (m, k)>,
    affine_map<(m, n, k) -> (k, n)>,
    affine_map<(m, n, k) -> (m, n)>
  ],
  iterator_types = ["parallel", "parallel", "reduction"]
}

// CHECK-DAG: #[[$mk:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$kn:.*]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[$mn:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @vectorization_test
func @vectorization_test(%A: memref<8x16xf32>, %B: memref<16x32xf32>,
                         %C: memref<8x32xf32>) {
  //       CHECK: vector.transfer_read %{{.*}} : memref<8x16xf32>, vector<8x16xf32>
  //       CHECK: vector.transfer_read %{{.*}} : memref<16x32xf32>, vector<16x32xf32>
  //       CHECK: vector.transfer_read %{{.*}} : memref<8x32xf32>, vector<8x32xf32>
  //       CHECK: vector.contract {indexing_maps = [#[[$mk]], #[[$kn]], #[[$mn]]]
  //  CHECK-SAME:   vector<8x16xf32>, vector<16x32xf32> into vector<8x32xf32>
  //       CHECK: vector.transfer_write %{{.*}}, %{{.*}} : vector<8x32xf32>, memref<8x32xf32>
  linalg.generic #matmul_trait
    ins(%A, %B : memref<8x16xf32>, memref<16x32xf32>)
   outs(%C : memref<8x32xf32>) {
    ^bb(%a: f32, %b: f32, %c: f32) :
      %d = mulf %a, %b: f32
      %e = addf %c, %d: f32
      linalg.yield %e : f32
  }
  return
}

// -----

#matmul_trait = {
  args_in = 2,
  args_out = 1,
  indexing_maps = [
    affine_map<(m, n, k) -> (m, k)>,
    affine_map<(m, n, k) -> (k, n)>,
    affine_map<(m, n, k) -> (m, n)>
  ],
  iterator_types = ["parallel", "parallel", "reduction"]
}

// CHECK-DAG: #[[$mk:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$kn:.*]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[$mn:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @vectorization_test_integer
func @vectorization_test_integer(%A: memref<8x16xi32>, %B: memref<16x32xi32>,
                                 %C: memref<8x32xi32>) {
  //       CHECK: vector.transfer_read %{{.*}} : memref<8x16xi32>, vector<8x16xi32>
  //       CHECK: vector.transfer_read %{{.*}} : memref<16x32xi32>, vector<16x32xi32>
  //       CHECK: vector.transfer_read %{{.*}} : memref<8x32xi32>, vector<8x32xi32>
  //       CHECK: vector.contract {indexing_maps = [#[[$mk]], #[[$kn]], #[[$mn]]],
  //  CHECK-SAME:   vector<8x16xi32>, vector<16x32xi32> into vector<8x32xi32>
  //       CHECK: vector.transfer_write %{{.*}}, %{{.*}} : vector<8x32xi32>, memref<8x32xi32>
  linalg.generic #matmul_trait
    ins(%A, %B : memref<8x16xi32>, memref<16x32xi32>)
   outs(%C : memref<8x32xi32>) {
    ^bb(%a: i32, %b: i32, %c: i32) :
      %d = muli %a, %b: i32
      %e = addi %c, %d: i32
      linalg.yield %e : i32
  }
  return
}

// -----

// CHECK-LABEL: func @vectorization_test_2
func @vectorization_test_2(%A: memref<8x16xf32>, %B: memref<16x32xf32>,
                         %C: memref<8x32xf32>) {
  //       CHECK: vector.contract {{.*}} :
  //                vector<8x16xf32>, vector<16x32xf32> into vector<8x32xf32>
  linalg.matmul
    ins(%A, %B: memref<8x16xf32>, memref<16x32xf32>)
   outs(%C: memref<8x32xf32>)
  return
}

// -----

// CHECK-LABEL: func @test_vectorize_fill
func @test_vectorize_fill(%A : memref<8x16xf32>, %arg0 : f32) {
  //       CHECK: %[[V:.*]] = vector.broadcast {{.*}} : f32 to vector<8x16xf32>
  //       CHECK: vector.transfer_write %[[V]], {{.*}} : vector<8x16xf32>, memref<8x16xf32>
  linalg.fill(%A, %arg0) :  memref<8x16xf32>, f32
  return
}

// -----

// CHECK-LABEL: func @test_vectorize_fill
func @test_vectorize_fill_scalar(%A : memref<f32>, %arg0 : f32) {
  //  CHECK-SAME: (%[[M:.*]]: memref<f32>, %[[V:.*]]: f32)
  //       CHECK:   store %[[V]], %[[M]][] : memref<f32>
  linalg.fill(%A, %arg0) :  memref<f32>, f32
  return
}

// -----

// CHECK-LABEL: func @test_vectorize_copy
func @test_vectorize_copy(%A : memref<8x16xf32>, %B : memref<8x16xf32>) {
  //       CHECK: %[[V:.*]] = vector.transfer_read {{.*}} : memref<8x16xf32>, vector<8x16xf32>
  //       CHECK: vector.transfer_write %[[V]], {{.*}} : vector<8x16xf32>, memref<8x16xf32>
  linalg.copy(%A, %B) :  memref<8x16xf32>, memref<8x16xf32>
  return
}

// -----

// CHECK-LABEL: func @test_vectorize_copy_scalar
func @test_vectorize_copy_scalar(%A : memref<f32>, %B : memref<f32>) {
  //       CHECK: %[[V:.*]] = memref.load {{.*}} : memref<f32>
  //       CHECK: store %[[V]], {{.*}} : memref<f32>
  linalg.copy(%A, %B) :  memref<f32>, memref<f32>
  return
}

// -----

// CHECK-LABEL: func @generic_vectorize
  //  CHECK-SAME: (%[[ARG0:.*]]: memref<4x256xf32>, %[[ARG1:.*]]: memref<4x256xf32>,
  //  CHECK-SAME:  %[[ARG2:.*]]: memref<256xf32>, %[[ARG3:.*]]: f32)
func @generic_vectorize(%arg0: memref<4x256xf32>,
                        %arg1: memref<4x256xf32>,
                        %arg2: memref<256xf32>, %i: f32) {
  //   CHECK-DAG:   %[[CST0:.*]] = constant dense<2.000000e+00> : vector<4x256xf32>
  //   CHECK-DAG:   %[[CST1:.*]] = constant dense<1.000000e+00> : vector<4x256xf32>
  //   CHECK-DAG:   %[[C0:.*]] = constant 0 : index
  %c1_f32 = constant 1.0 : f32
  linalg.generic {
    args_in = 0 : i64,
    args_out = 10 : i64,
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
  ins(%arg1, %arg2: memref<4x256xf32>, memref<256xf32>)
  outs(
    %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0 :
    memref<4x256xf32>, memref<4x256xf32>, memref<4x256xf32>, memref<4x256xf32>,
    memref<4x256xf32>, memref<4x256xf32>, memref<4x256xf32>, memref<4x256xf32>,
    memref<4x256xf32>, memref<4x256xf32>) {
  ^bb0(%arg3 : f32, %arg4 : f32, %arg5: f32, %arg6: f32, %arg7: f32, %arg8: f32,
  //       CHECK:   %[[V2:.*]] = vector.transfer_read %[[ARG1]][%[[C0]], %[[C0]]], {{.*}} : memref<4x256xf32>, vector<4x256xf32>
  //       CHECK:   %[[V0:.*]] = vector.transfer_read %[[ARG2]][%[[C0]]], {{.*}} : memref<256xf32>, vector<256xf32>
  //       CHECK:   %[[V3:.*]] = vector.transfer_read %[[ARG0]][%[[C0]], %[[C0]]], {{.*}} : memref<4x256xf32>, vector<4x256xf32>
  //       CHECK:   %[[V1:.*]] = vector.transfer_read %[[ARG0]][%[[C0]], %[[C0]]], {{.*}} : memref<4x256xf32>, vector<4x256xf32>
    %arg9 : f32, %arg10 : f32, %arg11 : f32, %arg12 : f32, %arg13 : f32,
    %arg14 : f32):
  //       CHECK:   %[[V0B:.*]] = vector.broadcast %[[V0]] : vector<256xf32> to vector<4x256xf32>
  //       CHECK:   %[[ADD:.*]] = addf %[[V0B]], %[[V1]] : vector<4x256xf32>
    %6 = addf %arg4, %arg6 : f32
  //       CHECK:   %[[CMP:.*]] = cmpf ogt, %[[V2]], %[[V1]] : vector<4x256xf32>
    %7 = cmpf ogt, %arg3, %arg6 : f32
  //       CHECK:   %[[ARG3B:.*]] = vector.broadcast %[[ARG3]] : f32 to vector<4x256xf32>
    %8 = constant 2.0 : f32
  //       CHECK:   %[[DIV:.*]] = divf %[[V3]], %[[ARG3B]] : vector<4x256xf32>
    %9 = divf %arg5, %i : f32
  //       CHECK:   %[[EXP:.*]] = math.exp2 %[[V3]] : vector<4x256xf32>
    %10 = math.exp2 %arg5 : f32
  //       CHECK:   %[[MUL:.*]] = mulf %[[V3]], %[[CST0]] : vector<4x256xf32>
    %11 = mulf %arg5, %8 : f32
  //       CHECK:   %[[RSQRT:.*]] = math.rsqrt %[[V3]] : vector<4x256xf32>
    %12 = math.rsqrt %arg5 : f32
  //       CHECK:   %[[SEL:.*]] = select %[[CMP]], %[[V3]], %[[V1]] : vector<4x256xi1>, vector<4x256xf32>
    %13 = select %7, %arg5, %arg6 : f32
  //       CHECK:   %[[V0B:.*]] = vector.broadcast %[[V0]] : vector<256xf32> to vector<4x256xf32>
  //       CHECK:   %[[SUB:.*]] = subf %[[V3]], %[[V0B]] : vector<4x256xf32>
    %14 = subf %arg5, %arg4 : f32
  //       CHECK:   %[[TAN:.*]] = math.tanh %[[V3]] : vector<4x256xf32>
    %15 = math.tanh %arg5 : f32
  //       CHECK:   vector.transfer_write %[[ADD]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, memref<4x256xf32>
  //       CHECK:   vector.transfer_write %[[CST0]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, memref<4x256xf32>
  //       CHECK:   vector.transfer_write %[[CST1]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, memref<4x256xf32>
  //       CHECK:   vector.transfer_write %[[DIV]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, memref<4x256xf32>
  //       CHECK:   vector.transfer_write %[[EXP]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, memref<4x256xf32>
  //       CHECK:   vector.transfer_write %[[MUL]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, memref<4x256xf32>
  //       CHECK:   vector.transfer_write %[[RSQRT]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, memref<4x256xf32>
  //       CHECK:   vector.transfer_write %[[SEL]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, memref<4x256xf32>
  //       CHECK:   vector.transfer_write %[[SUB]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, memref<4x256xf32>
  //       CHECK:   vector.transfer_write %[[TAN]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, memref<4x256xf32>
    linalg.yield %6, %8, %c1_f32, %9, %10, %11, %12, %13, %14, %15 : f32, f32,
      f32, f32, f32, f32, f32, f32, f32, f32
  }
  return
}


// -----

// CHECK-LABEL: func @generic_vectorize_tensor
//  CHECK-SAME: (%[[ARG0:.*]]: tensor<4x256xf32>, %[[ARG1:.*]]: tensor<4x256xf32>,
//  CHECK-SAME:  %[[ARG2:.*]]: tensor<256xf32>, %[[ARG3:.*]]: f32)
func @generic_vectorize_tensor(%arg0: tensor<4x256xf32>,
  %arg1: tensor<4x256xf32>, %arg2: tensor<256xf32>,
  %i: f32) -> (tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>,
    tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>,
    tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>) {
  %c1_f32 = constant 1.0 : f32
  %r:10 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
  ins(%arg1, %arg2: tensor<4x256xf32>, tensor<256xf32>)
  outs(
    %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0 :
    tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>,
    tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>,
    tensor<4x256xf32>, tensor<4x256xf32>) {
  ^bb0(%arg3 : f32, %arg4 : f32, %arg5: f32, %arg6: f32, %arg7: f32, %arg8: f32,
    %arg9 : f32, %arg10 : f32, %arg11 : f32, %arg12 : f32, %arg13 : f32,
    %arg14 : f32):
  //   CHECK-DAG:   %[[CST0:.*]] = constant dense<2.000000e+00> : vector<4x256xf32>
  //   CHECK-DAG:   %[[CST1:.*]] = constant dense<1.000000e+00> : vector<4x256xf32>
  //   CHECK-DAG:   %[[C0:.*]] = constant 0 : index
  //       CHECK:   %[[V2:.*]] = vector.transfer_read %[[ARG1]][%[[C0]], %[[C0]]], {{.*}} : tensor<4x256xf32>, vector<4x256xf32>
  //       CHECK:   %[[V0:.*]] = vector.transfer_read %[[ARG2]][%[[C0]]], {{.*}} : tensor<256xf32>, vector<256xf32>
  //       CHECK:   %[[V3:.*]] = vector.transfer_read %[[ARG0]][%[[C0]], %[[C0]]], {{.*}} : tensor<4x256xf32>, vector<4x256xf32>
  //       CHECK:   %[[V1:.*]] = vector.transfer_read %[[ARG0]][%[[C0]], %[[C0]]], {{.*}} : tensor<4x256xf32>, vector<4x256xf32>
  //       CHECK:   %[[V0B:.*]] = vector.broadcast %[[V0]] : vector<256xf32> to vector<4x256xf32>
  //       CHECK:   %[[ADD:.*]] = addf %[[V0B]], %[[V1]] : vector<4x256xf32>
    %6 = addf %arg4, %arg6 : f32
  //       CHECK:   %[[CMP:.*]] = cmpf ogt, %[[V2]], %[[V1]] : vector<4x256xf32>
    %7 = cmpf ogt, %arg3, %arg6 : f32
  //       CHECK:   %[[ARG3B:.*]] = vector.broadcast %[[ARG3]] : f32 to vector<4x256xf32>
    %8 = constant 2.0 : f32
  //       CHECK:   %[[DIV:.*]] = divf %[[V3]], %[[ARG3B]] : vector<4x256xf32>
    %9 = divf %arg5, %i : f32
  //       CHECK:   %[[EXP:.*]] = math.exp2 %[[V3]] : vector<4x256xf32>
    %10 = math.exp2 %arg5 : f32
  //       CHECK:   %[[MUL:.*]] = mulf %[[V3]], %[[CST0]] : vector<4x256xf32>
    %11 = mulf %arg5, %8 : f32
  //       CHECK:   %[[RSQRT:.*]] = math.rsqrt %[[V3]] : vector<4x256xf32>
    %12 = math.rsqrt %arg5 : f32
  //       CHECK:   %[[SEL:.*]] = select %[[CMP]], %[[V3]], %[[V1]] : vector<4x256xi1>, vector<4x256xf32>
    %13 = select %7, %arg5, %arg6 : f32
  //       CHECK:   %[[V0B:.*]] = vector.broadcast %[[V0]] : vector<256xf32> to vector<4x256xf32>
  //       CHECK:   %[[SUB:.*]] = subf %[[V3]], %[[V0B]] : vector<4x256xf32>
    %14 = subf %arg5, %arg4 : f32
  //       CHECK:   %[[TAN:.*]] = math.tanh %[[V3]] : vector<4x256xf32>
    %15 = math.tanh %arg5 : f32
  //       CHECK:   %[[R0:.*]] = vector.transfer_write %[[ADD]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, tensor<4x256xf32>
  //       CHECK:   %[[R1:.*]] = vector.transfer_write %[[CST0]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, tensor<4x256xf32>
  //       CHECK:   %[[R2:.*]] = vector.transfer_write %[[CST1]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, tensor<4x256xf32>
  //       CHECK:   %[[R3:.*]] = vector.transfer_write %[[DIV]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, tensor<4x256xf32>
  //       CHECK:   %[[R4:.*]] = vector.transfer_write %[[EXP]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, tensor<4x256xf32>
  //       CHECK:   %[[R5:.*]] = vector.transfer_write %[[MUL]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, tensor<4x256xf32>
  //       CHECK:   %[[R6:.*]] = vector.transfer_write %[[RSQRT]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, tensor<4x256xf32>
  //       CHECK:   %[[R7:.*]] = vector.transfer_write %[[SEL]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, tensor<4x256xf32>
  //       CHECK:   %[[R8:.*]] = vector.transfer_write %[[SUB]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, tensor<4x256xf32>
  //       CHECK:   %[[R9:.*]] = vector.transfer_write %[[TAN]], %[[ARG0]][%[[C0]], %[[C0]]] {{.*}} : vector<4x256xf32>, tensor<4x256xf32>
    linalg.yield %6, %8, %c1_f32, %9, %10, %11, %12, %13, %14, %15 : f32, f32,
      f32, f32, f32, f32, f32, f32, f32, f32
  } -> tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>,
    tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>,
    tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>
  //       CHECK:   return %[[R0]], %[[R1]], %[[R2]], %[[R3]], %[[R4]], %[[R5]], %[[R6]], %[[R7]], %[[R8]], %[[R9]] : tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>
  return %r#0, %r#1, %r#2, %r#3, %r#4, %r#5, %r#6, %r#7, %r#8, %r#9:
    tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>,
    tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>,
    tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>
}

// -----

// CHECK-LABEL: func @matmul_tensors
//  CHECK-SAME: (%[[ARG0:.*]]: tensor<8x4xf32>, %[[ARG1:.*]]: tensor<4x12xf32>,
//  CHECK-SAME:  %[[ARG2:.*]]: tensor<8x12xf32>) -> tensor<8x12xf32>
func @matmul_tensors(
  %arg0: tensor<8x4xf32>, %arg1: tensor<4x12xf32>, %arg2: tensor<8x12xf32>)
    -> tensor<8x12xf32> {
  //   CHECK-DAG:   %[[C0:.*]] = constant 0 : index
  //   CHECK-DAG:   %[[VEC_C0:.*]] = constant dense<0.000000e+00> : vector<8x12xf32>
  //   CHECK-DAG:   %[[V0:.*]] = vector.transfer_read %[[ARG0]][%[[C0]], %[[C0]]], {{.*}} : tensor<8x4xf32>, vector<8x4xf32>
  //   CHECK-DAG:   %[[V1:.*]] = vector.transfer_read %[[ARG1]][%[[C0]], %[[C0]]], {{.*}} : tensor<4x12xf32>, vector<4x12xf32>
  //   CHECK-DAG:   %[[V2:.*]] = vector.transfer_read %[[ARG2]][%[[C0]], %[[C0]]], {{.*}} : tensor<8x12xf32>, vector<8x12xf32>
  //
  // linalg contraction lowers to %tmp = vector.contract %a, %b, %c0 followed by addf %c, %tmp.
  // a later canonicalization fuses the add into vector.contract.
  //       CHECK:   %[[C:.*]] = vector.contract {{.*}} iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[V0]], %[[V1]], %[[VEC_C0]] : vector<8x4xf32>, vector<4x12xf32> into vector<8x12xf32>
  //       CHECK:   %[[C2:.*]] = addf %[[V2]], %[[C]] : vector<8x12xf32>
  //       CHECK:   %[[W:.*]] = vector.transfer_write %[[C2]], %[[ARG2]][%[[C0]], %[[C0]]] {masked = [false, false]} : vector<8x12xf32>, tensor<8x12xf32>
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<8x4xf32>, tensor<4x12xf32>)
                     outs(%arg2: tensor<8x12xf32>)
    -> tensor<8x12xf32>
  //       CHECK:   return %[[W]] : tensor<8x12xf32>
  return %0 : tensor<8x12xf32>
}

// -----

// CHECK-LABEL: func @matmul_i8_i8_i32
//  CHECK-SAME:  %[[ARG0:[a-z0-9]+]]: memref<4x6xi8>
//  CHECK-SAME:  %[[ARG1:[a-z0-9]+]]: memref<6x12xi8>
//  CHECK-SAME:  %[[ARG2:[a-z0-9]+]]: memref<4x12xi32>
func @matmul_i8_i8_i32(%a: memref<4x6xi8>, %b: memref<6x12xi8>, %c: memref<4x12xi32>) {
  //   CHECK-DAG:   %[[C0:.*]] = constant 0 : index
  //   CHECK-DAG:   %[[VEC_C0:.*]] = constant dense<0> : vector<4x12xi32>
  //   CHECK-DAG:   %[[V0:.*]] = vector.transfer_read %[[ARG0]][%[[C0]], %[[C0]]], {{.*}} : memref<4x6xi8>, vector<4x6xi8>
  //   CHECK-DAG:   %[[V1:.*]] = vector.transfer_read %[[ARG1]][%[[C0]], %[[C0]]], {{.*}} : memref<6x12xi8>, vector<6x12xi8>
  //   CHECK-DAG:   %[[V2:.*]] = vector.transfer_read %[[ARG2]][%[[C0]], %[[C0]]], {{.*}} : memref<4x12xi32>, vector<4x12xi32>
  //   CHECK-DAG:   %[[V0_32:.*]] = sexti %[[V0]] : vector<4x6xi8> to vector<4x6xi32>
  //   CHECK-DAG:   %[[V1_32:.*]] = sexti %[[V1]] : vector<6x12xi8> to vector<6x12xi32>
  //
  // linalg contraction lowers to %tmp = vector.contract %a, %b, %c0 followed by addf %c, %tmp.
  // a later canonicalization fuses the add into vector.contract.
  //       CHECK:   %[[C:.*]] = vector.contract {{.*}} iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[V0_32]], %[[V1_32]], %[[VEC_C0]]
  //  CHECK-SAME:     vector<4x6xi32>, vector<6x12xi32> into vector<4x12xi32>
  //       CHECK:   %[[RES:.*]] = addi %[[V2]], %[[C]] : vector<4x12xi32>
  //       CHECK:   vector.transfer_write %[[RES]], %[[ARG2]][%[[C0]], %[[C0]]] {masked = [false, false]}
  //  CHECK-SAME:     vector<4x12xi32>, memref<4x12xi32>
  linalg.matmul_i8_i8_i32 ins(%a, %b : memref<4x6xi8>, memref<6x12xi8>)
    outs(%c: memref<4x12xi32>)
  return
}

// -----

// CHECK-LABEL: func @pad_static
//   CHECK-NOT:   linalg.pad_tensor
func @pad_static(%arg0: tensor<?x?x?xf32>, %pad_value: f32) -> tensor<2x3x4xf32> {
  //      CHECK: %[[C0:.*]] = constant 0 : index
  //      CHECK: %[[READ:.*]] = vector.transfer_read %{{.*}}[%[[C0]], %[[C0]], %[[C0]]]
  // CHECK-SAME:   : tensor<?x?x?xf32>, vector<2x3x4xf32>
  //      CHECK: %[[INIT:.*]] = linalg.init_tensor [2, 3, 4] : tensor<2x3x4xf32>
  //      CHECK: %[[WRITTEN:.*]] = vector.transfer_write %[[READ]], %[[INIT]][%[[C0]], %[[C0]], %[[C0]]]
  // CHECK-SAME:   {masked = [false, false, false]} : vector<2x3x4xf32>, tensor<2x3x4xf32>
  %c0 = constant 0 : index
  %0 = linalg.pad_tensor %arg0 low[0, %c0, 0] high[0, 0, %c0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index):
      linalg.yield %pad_value : f32
    } : tensor<?x?x?xf32> to tensor<2x3x4xf32>

  // CHECK: return %[[WRITTEN]] : tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
}

// CHECK-LABEL: func @pad_static_high_padding
//       CHECK:   linalg.pad_tensor
func @pad_static_high_padding(%arg0: tensor<?x?x?xf32>, %pad_value: f32) -> tensor<2x3x4xf32> {
  %0 = linalg.pad_tensor %arg0 low[0, 0, 0] high[0, 1, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index):
      linalg.yield %pad_value : f32
    } : tensor<?x?x?xf32> to tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
}

// CHECK-LABEL: func @pad_dynamic
//       CHECK:   linalg.pad_tensor
func @pad_dynamic(%arg0: tensor<1x2x2x?xf32>, %low: index, %high: index,
                  %pad_value: f32) -> tensor<6x?x?x?xf32> {
  %0 = linalg.pad_tensor %arg0 low[2, %low, 3, 3] high[3, 3, %high, 2] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      linalg.yield %pad_value : f32
    } : tensor<1x2x2x?xf32> to tensor<6x?x?x?xf32>
  return %0 : tensor<6x?x?x?xf32>
}
