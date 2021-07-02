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

// CHECK-DAG: #[[$trans_2d:.*]] =  affine_map<(d0, d1) -> (d1, d0)>
// CHECK-DAG: #[[$mk:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$nk:.*]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG: #[[$mn:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @vectorization_test
func @vectorization_test(%A: memref<8x16xf32>, %B: memref<16x32xf32>,
                         %C: memref<8x32xf32>) {
  //       CHECK: vector.transfer_read %{{.*}} : memref<8x16xf32>, vector<8x16xf32>
  //       CHECK: vector.transfer_read %{{.*}} : memref<16x32xf32>, vector<32x16xf32>
  //       CHECK: vector.transfer_read %{{.*}} : memref<8x32xf32>, vector<8x32xf32>
  //       CHECK: vector.contract {indexing_maps = [#[[$mk]], #[[$nk]], #[[$mn]]]
  //  CHECK-SAME:   vector<8x16xf32>, vector<32x16xf32> into vector<8x32xf32>
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

// CHECK-DAG: #[[$trans_2d:.*]] =  affine_map<(d0, d1) -> (d1, d0)>
// CHECK-DAG: #[[$mk:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$nk:.*]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG: #[[$mn:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @vectorization_test_integer
func @vectorization_test_integer(%A: memref<8x16xi32>, %B: memref<16x32xi32>,
                                 %C: memref<8x32xi32>) {
  //       CHECK: vector.transfer_read %{{.*}} : memref<8x16xi32>, vector<8x16xi32>
  //       CHECK: vector.transfer_read %{{.*}} : memref<16x32xi32>, vector<32x16xi32>
  //       CHECK: vector.transfer_read %{{.*}} : memref<8x32xi32>, vector<8x32xi32>
  //       CHECK: vector.contract {indexing_maps = [#[[$mk]], #[[$nk]], #[[$mn]]],
  //  CHECK-SAME:   vector<8x16xi32>, vector<32x16xi32> into vector<8x32xi32>
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

// CHECK-LABEL: func @test_vectorize_scalar_input
func @test_vectorize_scalar_input(%A : memref<8x16xf32>, %arg0 : f32) {
  //       CHECK: %[[V:.*]] = vector.broadcast {{.*}} : f32 to vector<8x16xf32>
  //       CHECK: vector.transfer_write %[[V]], {{.*}} : vector<8x16xf32>, memref<8x16xf32>
  linalg.generic {
    indexing_maps = [affine_map<(m, n) -> ()>, affine_map<(m, n) -> (m, n)>],
    iterator_types = ["parallel", "parallel"]}
   ins(%arg0 : f32)
  outs(%A: memref<8x16xf32>) {
    ^bb(%0: f32, %1: f32) :
      linalg.yield %0 : f32
  }
  return
}

// -----

// CHECK-LABEL: func @test_vectorize_fill
func @test_vectorize_fill(%A : memref<8x16xf32>, %arg0 : f32) {
  //       CHECK: %[[V:.*]] = vector.broadcast {{.*}} : f32 to vector<8x16xf32>
  //       CHECK: vector.transfer_write %[[V]], {{.*}} : vector<8x16xf32>, memref<8x16xf32>
  linalg.fill(%arg0, %A) : f32, memref<8x16xf32>
  return
}

// -----

// CHECK-LABEL: func @test_vectorize_fill
func @test_vectorize_fill_scalar(%A : memref<f32>, %arg0 : f32) {
  //  CHECK-SAME: (%[[M:.*]]: memref<f32>, %[[V:.*]]: f32)
  //       CHECK:   store %[[V]], %[[M]][] : memref<f32>
  linalg.fill(%arg0, %A) : f32, memref<f32>
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

// CHECK-LABEL: func @test_vectorize_trailing_index
  //  CHECK-SAME: (%[[ARG0:.*]]: memref<1x2x4x8xindex>)
func @test_vectorize_trailing_index(%arg0: memref<1x2x4x8xindex>) {
  //   CHECK-DAG:   %[[CST0:.*]] = constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : vector<8xindex>
  //   CHECK-DAG:   %[[C0:.*]] = constant 0 : index
  linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
  outs(%arg0: memref<1x2x4x8xindex>) {
  ^bb0(%arg1: index):
  //       CHECK:   %[[BCST:.*]] = vector.broadcast %[[CST0]] : vector<8xindex> to vector<1x2x4x8xindex>
  //       CHECK:   vector.transfer_write %[[BCST]], %[[ARG0]][%[[C0]], %[[C0]], %[[C0]], %[[C0]]] {{.*}} : vector<1x2x4x8xindex>, memref<1x2x4x8xindex>
    %0 = linalg.index 3 : index
    linalg.yield %0 : index
  }
  return
}

// -----

// CHECK-LABEL: func @test_vectorize_inner_index
  //  CHECK-SAME: (%[[ARG0:.*]]: memref<1x2x4x8xindex>)
func @test_vectorize_inner_index(%arg0: memref<1x2x4x8xindex>) {
  //   CHECK-DAG:   %[[CST0:.*]] = constant dense<[0, 1]> : vector<2xindex>
  //   CHECK-DAG:   %[[C0:.*]] = constant 0 : index
  linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
  outs(%arg0: memref<1x2x4x8xindex>) {
  ^bb0(%arg1: index):
  //       CHECK:   %[[BCST:.*]] = vector.broadcast %[[CST0]] : vector<2xindex> to vector<1x8x4x2xindex>
  //       CHECK:   %[[TRAN:.*]] = vector.transpose %[[BCST]], [0, 3, 2, 1] : vector<1x8x4x2xindex> to vector<1x2x4x8xindex>
  //       CHECK:   vector.transfer_write %[[TRAN]], %[[ARG0]][%[[C0]], %[[C0]], %[[C0]], %[[C0]]] {{.*}} : vector<1x2x4x8xindex>, memref<1x2x4x8xindex>
    %0 = linalg.index 1 : index
    linalg.yield %0 : index
  }
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
  //       CHECK:   %[[V0:.*]] = vector.transfer_read %[[ARG2]][%[[C0]]], {{.*}} : memref<256xf32>, vector<4x256xf32>
  //       CHECK:   %[[V3:.*]] = vector.transfer_read %[[ARG0]][%[[C0]], %[[C0]]], {{.*}} : memref<4x256xf32>, vector<4x256xf32>
  //       CHECK:   %[[V1:.*]] = vector.transfer_read %[[ARG0]][%[[C0]], %[[C0]]], {{.*}} : memref<4x256xf32>, vector<4x256xf32>
    %arg9 : f32, %arg10 : f32, %arg11 : f32, %arg12 : f32, %arg13 : f32,
    %arg14 : f32):
  //       CHECK:   %[[ADD:.*]] = addf %[[V0]], %[[V1]] : vector<4x256xf32>
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
  //       CHECK:   %[[SUB:.*]] = subf %[[V3]], %[[V0]] : vector<4x256xf32>
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
  //       CHECK:   %[[V0:.*]] = vector.transfer_read %[[ARG2]][%[[C0]]], {{.*}} : tensor<256xf32>, vector<4x256xf32>
  //       CHECK:   %[[V3:.*]] = vector.transfer_read %[[ARG0]][%[[C0]], %[[C0]]], {{.*}} : tensor<4x256xf32>, vector<4x256xf32>
  //       CHECK:   %[[V1:.*]] = vector.transfer_read %[[ARG0]][%[[C0]], %[[C0]]], {{.*}} : tensor<4x256xf32>, vector<4x256xf32>
  //       CHECK:   %[[ADD:.*]] = addf %[[V0]], %[[V1]] : vector<4x256xf32>
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
  //       CHECK:   %[[SUB:.*]] = subf %[[V3]], %[[V0]] : vector<4x256xf32>
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
  } -> (tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>,
    tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>,
    tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>)
  //       CHECK:   return %[[R0]], %[[R1]], %[[R2]], %[[R3]], %[[R4]], %[[R5]], %[[R6]], %[[R7]], %[[R8]], %[[R9]] : tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>
  return %r#0, %r#1, %r#2, %r#3, %r#4, %r#5, %r#6, %r#7, %r#8, %r#9:
    tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>,
    tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>,
    tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>
}

// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, 0, 0, d1)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0) -> (d0, 0, 0, 0)>
// CHECK-DAG: #[[$MAP2:.*]] = affine_map<(d0) -> (0, 0, d0, 0)>
// CHECK-DAG: #[[$MAP3:.*]] = affine_map<(d0, d1) -> (d1, 0, d0, 0)>
//     CHECK: func @generic_vectorize_broadcast_transpose
// CHECK-DAG:   %[[C0:.*]] = constant 0 : index
// CHECK-DAG:   %[[CF:.*]] = constant 0.000000e+00 : f32
//     CHECK:   %[[V0:.*]] = vector.transfer_read %{{.*}}[%[[C0]], %[[C0]]], %[[CF]] {in_bounds = [true, true, true, true], permutation_map = #[[$MAP0]]} : memref<4x4xf32>, vector<4x4x4x4xf32>
//     CHECK:   %[[V1:.*]] = vector.transfer_read %{{.*}}[%[[C0]]], %[[CF]] {in_bounds = [true, true, true, true], permutation_map = #[[$MAP1]]} : memref<4xf32>, vector<4x4x4x4xf32>
//     CHECK:   %[[V2:.*]] = vector.transfer_read %{{.*}}[%[[C0]]], %[[CF]] {in_bounds = [true, true, true, true], permutation_map = #[[$MAP2]]} : memref<4xf32>, vector<4x4x4x4xf32>
//     CHECK:   %[[V3:.*]] = vector.transfer_read %{{.*}}[%[[C0]], %[[C0]]], %[[CF]] {in_bounds = [true, true, true, true], permutation_map = #[[$MAP3]]} : memref<4x4xf32>, vector<4x4x4x4xf32>
//     CHECK:   %[[SUB:.*]] = subf %[[V0]], %[[V1]] : vector<4x4x4x4xf32>
//     CHECK:   %[[ADD0:.*]] = addf %[[V2]], %[[SUB]] : vector<4x4x4x4xf32>
//     CHECK:   %[[ADD1:.*]] = addf %[[V3]], %[[ADD0]] : vector<4x4x4x4xf32>
//     CHECK: vector.transfer_write %[[ADD1]], {{.*}} : vector<4x4x4x4xf32>, memref<4x4x4x4xf32>
func @generic_vectorize_broadcast_transpose(
  %A: memref<4xf32>, %B: memref<4x4xf32>, %C: memref<4x4x4x4xf32>) {
  linalg.generic {
  indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3)>,
                   affine_map<(d0, d1, d2, d3) -> (d0)>,
                   affine_map<(d0, d1, d2, d3) -> (d2)>,
                   affine_map<(d0, d1, d2, d3) -> (d2, d0)>,
                   affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
  iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
  ins(%B, %A, %A, %B: memref<4x4xf32>, memref<4xf32>, memref<4xf32>, memref<4x4xf32>)
  outs(%C : memref<4x4x4x4xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
    %s = subf %arg0, %arg1 : f32
    %a = addf %arg2, %s : f32
    %b = addf %arg3, %a : f32
    linalg.yield %b : f32
  }
  return
}

// -----

// Test different input maps.
#matmul_trait = {
  indexing_maps = [
    affine_map<(d0, d1, d2, d3) -> (d1, d0)>,
    affine_map<(d0, d1, d2, d3) -> (d3, d1)>,
    affine_map<(d0, d1, d2, d3) -> (d3, d1, d0, d2)>,
    affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  ],
  iterator_types = ["parallel", "parallel", "parallel", "parallel"]
}

// CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0, d1) -> (d1, d0, 0, 0)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1) -> (0, d1, 0, d0)>
// CHECK-DAG: #[[MAP2:.*]] = affine_map<(d0, d1, d2, d3) -> (d2, d1, d3, d0)>
//       CHECK: func @vectorization_transpose
//       CHECK: vector.transfer_read {{.*}}{in_bounds = [true, true, true, true], permutation_map = #[[MAP0]]} : memref<14x7xf32>, vector<7x14x8x16xf32>
//       CHECK: vector.transfer_read {{.*}}{in_bounds = [true, true, true, true], permutation_map = #[[MAP1]]} : memref<16x14xf32>, vector<7x14x8x16xf32>
//       CHECK: vector.transfer_read {{.*}}{in_bounds = [true, true, true, true], permutation_map = #[[MAP2]]} : memref<16x14x7x8xf32>, vector<7x14x8x16xf32>
//       CHECK: addf {{.*}} : vector<7x14x8x16xf32>
//       CHECK: addf {{.*}} : vector<7x14x8x16xf32>
//       CHECK: vector.transfer_write {{.*}} : vector<7x14x8x16xf32>, memref<7x14x8x16xf32>
func @vectorization_transpose(%A: memref<14x7xf32>, %B: memref<16x14xf32>,
                         %C: memref<16x14x7x8xf32>, %D: memref<7x14x8x16xf32>) {
  linalg.generic #matmul_trait
    ins(%A, %B, %C : memref<14x7xf32>, memref<16x14xf32>, memref<16x14x7x8xf32>)
   outs(%D : memref<7x14x8x16xf32>) {
    ^bb(%a: f32, %b: f32, %c: f32, %d: f32) :
      %e = addf %a, %b: f32
      %f = addf %e, %c: f32
      linalg.yield %f : f32
  }
  return
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
  //   CHECK-DAG:   %[[V1:.*]] = vector.transfer_read %[[ARG1]][%[[C0]], %[[C0]]], {{.*}} : tensor<4x12xf32>, vector<12x4xf32>
  //   CHECK-DAG:   %[[V2:.*]] = vector.transfer_read %[[ARG2]][%[[C0]], %[[C0]]], {{.*}} : tensor<8x12xf32>, vector<8x12xf32>
  //
  // linalg contraction lowers to %tmp = vector.contract %a, %b, %c0 followed by addf %c, %tmp.
  // a later canonicalization fuses the add into vector.contract.
  //       CHECK:   %[[C:.*]] = vector.contract
  //  CHECK-SAME:     iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>}
  //  CHECK-SAME:     %[[V0]], %[[V1]], %[[VEC_C0]] :
  //  CHECK-SAME:     vector<8x4xf32>, vector<12x4xf32> into vector<8x12xf32>
  //       CHECK:   %[[C2:.*]] = addf %[[V2]], %[[C]] : vector<8x12xf32>
  //       CHECK:   %[[W:.*]] = vector.transfer_write %[[C2]], %[[ARG2]][%[[C0]], %[[C0]]] {in_bounds = [true, true]} : vector<8x12xf32>, tensor<8x12xf32>
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
  //   CHECK-DAG:   %[[V1:.*]] = vector.transfer_read %[[ARG1]][%[[C0]], %[[C0]]], {{.*}} : memref<6x12xi8>, vector<12x6xi8>
  //   CHECK-DAG:   %[[V2:.*]] = vector.transfer_read %[[ARG2]][%[[C0]], %[[C0]]], {{.*}} : memref<4x12xi32>, vector<4x12xi32>
  //   CHECK-DAG:   %[[V0_32:.*]] = sexti %[[V0]] : vector<4x6xi8> to vector<4x6xi32>
  //   CHECK-DAG:   %[[V1_32:.*]] = sexti %[[V1]] : vector<12x6xi8> to vector<12x6xi32>
  //
  // linalg contraction lowers to %tmp = vector.contract %a, %b, %c0 followed by addf %c, %tmp.
  // a later canonicalization fuses the add into vector.contract.
  //       CHECK:   %[[C:.*]] = vector.contract
  //  CHECK-SAME:      iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>}
  //  CHECK-SAME:      %[[V0_32]], %[[V1_32]], %[[VEC_C0]]
  //  CHECK-SAME:     vector<4x6xi32>, vector<12x6xi32> into vector<4x12xi32>
  //       CHECK:   %[[RES:.*]] = addi %[[V2]], %[[C]] : vector<4x12xi32>
  //       CHECK:   vector.transfer_write %[[RES]], %[[ARG2]][%[[C0]], %[[C0]]] {in_bounds = [true, true]}
  //  CHECK-SAME:     vector<4x12xi32>, memref<4x12xi32>
  linalg.matmul_i8_i8_i32 ins(%a, %b : memref<4x6xi8>, memref<6x12xi8>)
    outs(%c: memref<4x12xi32>)
  return
}

// -----

// CHECK-LABEL: func @pad_static(
//  CHECK-SAME:                  %[[ARG0:.*]]: tensor<2x?x2xf32>, %[[PAD:.*]]: f32
//   CHECK-NOT:   linalg.pad_tensor
//   CHECK-DAG:   %[[C0:.*]] = constant 0 : index
//   CHECK-DAG:   %[[C2:.*]] = constant 2 : index
//   CHECK-DAG:   %[[INIT:.*]] = linalg.init_tensor [2, 3, 4] : tensor<2x3x4xf32>
//   CHECK-DAG:   %[[VEC:.*]] = vector.broadcast %[[PAD]] : f32 to vector<2x3x4xf32>
//       CHECK:   %[[FILL:.*]] = vector.transfer_write %[[VEC]], %[[INIT]]{{.*}} : vector<2x3x4xf32>, tensor<2x3x4xf32>
//       CHECK:   %[[READ:.*]] = vector.transfer_read %[[ARG0]][%[[C0]], %[[C0]], %[[C0]]], %[[PAD]] {in_bounds = [true, false, true]} : tensor<2x?x2xf32>, vector<2x3x2xf32>
//       CHECK:   %[[RESULT:.*]] = vector.transfer_write %[[READ]], %[[FILL]][%[[C0]], %[[C0]], %[[C2]]] {in_bounds = [true, true, true]} : vector<2x3x2xf32>, tensor<2x3x4xf32>
//       CHECK:   return %[[RESULT]]
func @pad_static(%arg0: tensor<2x?x2xf32>, %pad_value: f32) -> tensor<2x3x4xf32> {
  %0 = linalg.pad_tensor %arg0 low[0, 0, 2] high[0, 1, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index):
      linalg.yield %pad_value : f32
    } : tensor<2x?x2xf32> to tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
}

// -----

// CHECK-LABEL: func @pad_static_source(
//  CHECK-SAME:                  %[[ARG0:.*]]: tensor<2x5x2xf32>, %[[PAD:.*]]: f32
//   CHECK-NOT:   linalg.pad_tensor
//   CHECK-DAG:   %[[C0:.*]] = constant 0 : index
//   CHECK-DAG:   %[[C2:.*]] = constant 2 : index
//       CHECK:   %[[INIT:.*]] = linalg.init_tensor [2, 6, 4] : tensor<2x6x4xf32>
//       CHECK:   %[[VEC:.*]] =  vector.broadcast %[[PAD]] : f32 to vector<2x6x4xf32>
//       CHECK:   %[[FILL:.*]] = vector.transfer_write %[[VEC]], %[[INIT]][%[[C0]], %[[C0]], %[[C0]]] {in_bounds = [true, true, true]} : vector<2x6x4xf32>, tensor<2x6x4xf32>
//       CHECK:   %[[READ:.*]] = vector.transfer_read %[[ARG0]][%[[C0]], %[[C0]], %[[C0]]], %{{.*}} {in_bounds = [true, true, true]} : tensor<2x5x2xf32>, vector<2x5x2xf32>
//       CHECK:   %[[WRITE:.*]] = vector.transfer_write %[[READ]], %[[FILL]][%[[C0]], %[[C0]], %[[C2]]] {in_bounds = [true, true, true]} : vector<2x5x2xf32>, tensor<2x6x4xf32>
//       CHECK:   return %[[WRITE]]
func @pad_static_source(%arg0: tensor<2x5x2xf32>, %pad_value: f32) -> tensor<2x6x4xf32> {
  %0 = linalg.pad_tensor %arg0 low[0, 0, 2] high[0, 1, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index):
      linalg.yield %pad_value : f32
    } : tensor<2x5x2xf32> to tensor<2x6x4xf32>
  return %0 : tensor<2x6x4xf32>
}

// -----

// CHECK-LABEL: func @pad_static_dynamic(
//  CHECK-SAME:                          %[[SRC:.*]]: tensor<1x2x2x?xf32>, %[[LOW:.*]]: index, %[[HIGH:.*]]: index
//   CHECK-NOT:   linalg.pad_tensor
//   CHECK-DAG:   %[[C2:.*]] = constant 2 : index
//   CHECK-DAG:   %[[C3:.*]] = constant 3 : index
//   CHECK-DAG:   %[[C5:.*]] = constant 5 : index
//       CHECK:   %[[V0:.*]] = addi %[[LOW]], %[[C2]] : index
//       CHECK:   %[[V1:.*]] = addi %[[V0]], %[[C3]] : index
//       CHECK:   %[[V2:.*]] = addi %[[HIGH]], %[[C5]] : index
//       CHECK:   %[[DIM3:.*]] = tensor.dim %[[SRC]], %[[C3]] : tensor<1x2x2x?xf32>
//       CHECK:   %[[V4:.*]] = addi %[[DIM3]], %[[C3]] : index
//       CHECK:   %[[V5:.*]] = addi %[[V4]], %[[C2]] : index
//       CHECK:   %[[INIT:.*]] = linalg.init_tensor [6, %[[V1]], %[[V2]], %[[V5]]] : tensor<6x?x?x?xf32>
//       CHECK:   %[[FILL:.*]] = linalg.fill(%{{.*}}, %[[INIT]]) : f32, tensor<6x?x?x?xf32> -> tensor<6x?x?x?xf32>
//       CHECK:   %[[SRCDIM:.*]] = tensor.dim %[[SRC]], %[[C3]] : tensor<1x2x2x?xf32>
//       CHECK:   %[[RESULT:.*]] = tensor.insert_slice %[[SRC]] into %[[FILL]][2, %[[LOW]], 3, 3] [1, 2, 2, %[[SRCDIM]]] [1, 1, 1, 1] : tensor<1x2x2x?xf32> into tensor<6x?x?x?xf32>
//       CHECK:   return %[[RESULT]]
func @pad_static_dynamic(%arg0: tensor<1x2x2x?xf32>, %low: index, %high: index,
                  %pad_value: f32) -> tensor<6x?x?x?xf32> {
  %0 = linalg.pad_tensor %arg0 low[2, %low, 3, 3] high[3, 3, %high, 2] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      linalg.yield %pad_value : f32
    } : tensor<1x2x2x?xf32> to tensor<6x?x?x?xf32>
  return %0 : tensor<6x?x?x?xf32>
}

// -----

// CHECK-LABEL: func @pad_and_transfer_read
//  CHECK-SAME:     %[[ARG0:.*]]: tensor<5x6xf32>
//   CHECK-NOT:   linalg.pad_tensor
//   CHECK-DAG:   %[[C0:.*]] = constant 0 : index
//   CHECK-DAG:   %[[C5:.*]] = constant 5.0
//       CHECK:   %[[RESULT:.*]] = vector.transfer_read %[[ARG0]][%[[C0]], %[[C0]]], %[[C5]] : tensor<5x6xf32>, vector<7x9xf32>
//       CHECK:   return %[[RESULT]]
func @pad_and_transfer_read(%arg0: tensor<5x6xf32>) -> vector<7x9xf32> {
  %c0 = constant 0 : index
  %c5 = constant 5.0 : f32
  %c6 = constant 6.0 : f32
  %0 = linalg.pad_tensor %arg0 low[0, 0] high[5, 7] {
    ^bb0(%arg1: index, %arg2: index):
      linalg.yield %c5 : f32
  } : tensor<5x6xf32> to tensor<10x13xf32>
  %1 = vector.transfer_read %0[%c0, %c0], %c6
      : tensor<10x13xf32>, vector<7x9xf32>
  return %1 : vector<7x9xf32>
}

// -----


// CHECK-LABEL: func @pad_and_transfer_write_static
//  CHECK-SAME:     %[[ARG0:.*]]: tensor<5x6xf32>, %[[ARG1:.*]]: vector<7x9xf32>
//   CHECK-NOT:   linalg.pad_tensor
//       CHECK:   %[[C0:.*]] = constant 0 : index
//       CHECK:   %[[RESULT:.*]] = vector.transfer_write %[[ARG1]], %[[ARG0]][%[[C0]], %[[C0]]] : vector<7x9xf32>, tensor<5x6xf32>
//       CHECK:   return %[[RESULT]]
func @pad_and_transfer_write_static(
    %arg0: tensor<5x6xf32>, %arg1: vector<7x9xf32>) -> tensor<5x6xf32> {
  %c0 = constant 0 : index
  %c5 = constant 5.0 : f32
  %0 = linalg.pad_tensor %arg0 low[0, 0] high[5, 7] {
    ^bb0(%arg2: index, %arg3: index):
      linalg.yield %c5 : f32
  } : tensor<5x6xf32> to tensor<10x13xf32>
  %1 = vector.transfer_write %arg1, %0[%c0, %c0]
      : vector<7x9xf32>, tensor<10x13xf32>
  %2 = tensor.extract_slice %1[0, 0] [5, 6] [1, 1] : tensor<10x13xf32> to tensor<5x6xf32>
  return %2 : tensor<5x6xf32>
}

// -----

// CHECK-LABEL: func @pad_and_transfer_write_dynamic_static
//  CHECK-SAME:     %[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: vector<7x9xf32>, %[[SIZE:.*]]: index, %[[PADDING:.*]]: index
//   CHECK-NOT:   linalg.pad_tensor
//       CHECK:   %[[C0:.*]] = constant 0 : index
//       CHECK:   %[[SUB:.*]] = tensor.extract_slice %[[ARG0]][0, 0] [%[[SIZE]], 6] [1, 1] : tensor<?x?xf32> to tensor<?x6xf32>
//       CHECK:   %[[RESULT:.*]] = vector.transfer_write %[[ARG1]], %[[SUB]][%[[C0]], %[[C0]]] : vector<7x9xf32>, tensor<?x6xf32>
//       CHECK:   return %[[RESULT]]
func @pad_and_transfer_write_dynamic_static(
    %arg0: tensor<?x?xf32>, %arg1: vector<7x9xf32>, %size: index, %padding: index) -> tensor<?x6xf32> {
  %c0 = constant 0 : index
  %c5 = constant 5.0 : f32
  %s = tensor.extract_slice %arg0[0, 0] [%size, 6] [1, 1]
      : tensor<?x?xf32> to tensor<?x6xf32>
  %0 = linalg.pad_tensor %s low[0, 0] high[%padding, 7] {
    ^bb0(%arg2: index, %arg3: index):
      linalg.yield %c5 : f32
  } : tensor<?x6xf32> to tensor<?x13xf32>
  %1 = vector.transfer_write %arg1, %0[%c0, %c0]
      : vector<7x9xf32>, tensor<?x13xf32>
  %2 = tensor.extract_slice %1[0, 0] [%size, 6] [1, 1] : tensor<?x13xf32> to tensor<?x6xf32>
  return %2 : tensor<?x6xf32>
}

// -----

// CHECK-LABEL: func @pad_and_insert_slice
//  CHECK-SAME:     %[[ARG0:.*]]: tensor<5x6xf32>, %[[ARG1:.*]]: tensor<12x13xf32>
//   CHECK-NOT:   linalg.pad_tensor
//   CHECK-DAG:   %[[C0:.*]] = constant 0 : index
//   CHECK-DAG:   %[[C5:.*]] = constant 5.0
//       CHECK:   %[[READ:.*]] = vector.transfer_read %[[ARG0]][%[[C0]], %[[C0]]], %[[C5]] : tensor<5x6xf32>, vector<7x9xf32>
//       CHECK:   %[[WRITE:.*]] = vector.transfer_write %[[READ]], %[[ARG1]][%[[C0]], %[[C0]]] {in_bounds = [true, true]} : vector<7x9xf32>, tensor<12x13xf32>
//       CHECK:   return %[[WRITE]]
func @pad_and_insert_slice(
    %arg0: tensor<5x6xf32>, %arg1: tensor<12x13xf32>) -> tensor<12x13xf32> {
  %c0 = constant 0 : index
  %c5 = constant 5.0 : f32
  %0 = linalg.pad_tensor %arg0 low[0, 0] high[2, 3] {
    ^bb0(%arg2: index, %arg3: index):
      linalg.yield %c5 : f32
  } : tensor<5x6xf32> to tensor<7x9xf32>
  %r = tensor.insert_slice %0 into %arg1[0, 0][7, 9][1, 1] : tensor<7x9xf32> into tensor<12x13xf32>
  return %r : tensor<12x13xf32>
}

// -----

// CHECK-LABEL: func @pad_tensor_non_const_pad_value
//  CHECK-SAME:     %[[ARG0:.*]]: tensor<5x6xf32>
//   CHECK-NOT:   linalg.pad_tensor
//   CHECK-DAG:   %[[C0:.*]] = constant 0 : index
//   CHECK-DAG:   %[[C3:.*]] = constant 3 : index
//   CHECK-DAG:   %[[C4:.*]] = constant 4 : index
//       CHECK:   %[[FILL:.*]] = tensor.generate
//       CHECK:     %[[RES:.*]] = mulf
//       CHECK:     tensor.yield %[[RES]] : f32
//       CHECK:   %[[READ:.*]] = vector.transfer_read %[[ARG0]][%[[C0]], %[[C0]]], %{{.*}} {in_bounds = [true, true]} : tensor<5x6xf32>, vector<5x6xf32>
//       CHECK:   %[[WRITE:.*]] = vector.transfer_write %[[READ]], %[[FILL]][%[[C3]], %[[C4]]] {in_bounds = [true, true]} : vector<5x6xf32>, tensor<12x13xf32>
//       CHECK:   return %[[WRITE]]
func @pad_tensor_non_const_pad_value(%arg0: tensor<5x6xf32>) -> tensor<12x13xf32> {
  %c0 = constant 0 : index
  %c5 = constant 5.0 : f32
  %0 = linalg.pad_tensor %arg0 low[3, 4] high[4, 3] {
    ^bb0(%arg1: index, %arg2: index):
      %i1 = index_cast %arg1 : index to i32
      %i2 = index_cast %arg2 : index to i32
      %f1 = sitofp %i1 : i32 to f32
      %f2 = sitofp %i2 : i32 to f32
      %m = mulf %f1, %f2 : f32
      linalg.yield %m : f32
  } : tensor<5x6xf32> to tensor<12x13xf32>
  return %0 : tensor<12x13xf32>
}

// -----

// CHECK-DAG: #[[$M0:.*]] = affine_map<(d0, d1) -> (d0, d1, 0)>

// CHECK-LABEL: func @sum_exp
func @sum_exp(%input: tensor<4x16x8xf32>, %output: tensor<4x16xf32>)
  -> tensor<4x16xf32>
{
  // CHECK: vector.transfer_read {{.*}} : tensor<4x16x8xf32>, vector<4x16x8xf32>
  // CHECK: vector.transfer_read {{.*}} {in_bounds = [true, true, true], permutation_map = #[[$M0]]} : tensor<4x16xf32>, vector<4x16x8xf32>
  // CHECK: math.exp {{.*}} : vector<4x16x8xf32>
  // CHECK: addf {{.*}} : vector<4x16x8xf32>
  // CHECK: vector.multi_reduction #vector.kind<add>, %{{.*}} [2] : vector<4x16x8xf32> to vector<4x16xf32>
  // CHECK: vector.transfer_write {{.*}} : vector<4x16xf32>, tensor<4x16xf32>
  // CHECK: return {{.*}} : tensor<4x16xf32>
  %0 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%input : tensor<4x16x8xf32>) outs(%output : tensor<4x16xf32>) {
    ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
      %1 = math.exp %arg0 : f32
      %2 = addf %1, %arg1 : f32
      linalg.yield %2 : f32
    } -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// -----

// CHECK-DAG: #[[$M1:.*]] =  affine_map<(d0, d1) -> (d1, d0, 0, 0)>
// CHECK-DAG: #[[$M2:.*]] =  affine_map<(d0, d1) -> (0, 0, d1, d0)>
// CHECK-DAG: #[[$M3:.*]] =  affine_map<(d0, d1) -> (d1, 0, 0, d0)>
// CHECK-DAG: #[[$M4:.*]] =  affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: func @sum_exp_2
func @sum_exp_2(%input: tensor<3x2xf32>, %input_2: tensor<5x4xf32>, %output: tensor<5x2xf32>)
  -> tensor<5x2xf32>
{
  // CHECK: vector.transfer_read {{.*}} {in_bounds = [true, true, true, true], permutation_map = #[[$M1]]} : tensor<3x2xf32>, vector<2x3x4x5xf32>
  // CHECK: vector.transfer_read {{.*}} {in_bounds = [true, true, true, true], permutation_map = #[[$M2]]} : tensor<5x4xf32>, vector<2x3x4x5xf32>
  // CHECK: vector.transfer_read {{.*}} {in_bounds = [true, true, true, true], permutation_map = #[[$M3]]} : tensor<5x2xf32>, vector<2x3x4x5xf32>
  // CHECK: math.exp {{.*}} : vector<2x3x4x5xf32>
  // CHECK: math.exp {{.*}} : vector<2x3x4x5xf32>
  // CHECK: addf {{.*}} : vector<2x3x4x5xf32>
  // CHECK: addf {{.*}} : vector<2x3x4x5xf32>
  // CHECK: vector.multi_reduction #vector.kind<add>, {{.*}}  [1, 2] : vector<2x3x4x5xf32> to vector<2x5xf32>
  // CHECK: vector.transfer_write {{.*}} {in_bounds = [true, true], permutation_map = #[[$M4]]} : vector<2x5xf32>, tensor<5x2xf32>
  // CHECK: return {{.*}} : tensor<5x2xf32>
  %0 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d1, d0)>,
        affine_map<(d0, d1, d2, d3) -> (d3, d2)>,
        affine_map<(d0, d1, d2, d3) -> (d3, d0)>
      ],
      iterator_types = ["parallel", "reduction", "reduction", "parallel"]
    } ins(%input, %input_2 : tensor<3x2xf32>, tensor<5x4xf32>) outs(%output : tensor<5x2xf32>) {
    ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):  // no predecessors
      %1 = math.exp %arg0 : f32
      %2 = math.exp %arg1 : f32
      %3 = addf %1, %2 : f32
      %4 = addf %3, %arg2 : f32
      linalg.yield %4 : f32
    } -> tensor<5x2xf32>
  return %0 : tensor<5x2xf32>
}
