// RUN: mlir-opt %s -test-linalg-transform-patterns=test-linalg-to-vector-patterns | FileCheck %s

// CHECK-DAG: #[[$mk:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$kn:.*]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[$mn:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: contraction_dot
func @contraction_dot(%A: memref<1584xf32>, %B: memref<1584xf32>, %C: memref<f32>) {
  // CHECK: vector.contract
  // CHECK-SAME: vector<1584xf32>, vector<1584xf32> into f32
  linalg.dot ins(%A, %B: memref<1584xf32>, memref<1584xf32>)
            outs(%C: memref<f32>)
  return
}

// CHECK-LABEL: contraction_matvec
func @contraction_matvec(%A: memref<1584x1584xf32>, %B: memref<1584xf32>, %C: memref<1584xf32>) {
  // CHECK: vector.contract
  // CHECK-SAME: vector<1584x1584xf32>, vector<1584xf32> into vector<1584xf32>
  linalg.matvec ins(%A, %B: memref<1584x1584xf32>, memref<1584xf32>)
            outs(%C: memref<1584xf32>)
  return
}

// CHECK-LABEL: contraction_matmul
func @contraction_matmul(%A: memref<1584x1584xf32>, %B: memref<1584x1584xf32>, %C: memref<1584x1584xf32>) {
  // CHECK: vector.contract
  // CHECK-SAME: vector<1584x1584xf32>, vector<1584x1584xf32> into vector<1584x1584xf32>
  linalg.matmul ins(%A, %B: memref<1584x1584xf32>, memref<1584x1584xf32>)
            outs(%C: memref<1584x1584xf32>)
  return
}

// CHECK-LABEL: contraction_batch_matmul
func @contraction_batch_matmul(%A: memref<1584x1584x1584xf32>, %B: memref<1584x1584x1584xf32>, %C: memref<1584x1584x1584xf32>) {
  // CHECK: vector.contract
  // CHECK-SAME: vector<1584x1584x1584xf32>, vector<1584x1584x1584xf32> into vector<1584x1584x1584xf32>
  linalg.batch_matmul
    ins(%A, %B: memref<1584x1584x1584xf32>, memref<1584x1584x1584xf32>)
   outs(%C: memref<1584x1584x1584xf32>)
  return
}

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
func @vectorization_test(%A: memref<8x16xf32>, %B: memref<16x32xf32>,
                         %C: memref<8x32xf32>) {
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
// CHECK-LABEL: func @vectorization_test
//       CHECK: vector.transfer_read %{{.*}} : memref<8x16xf32>, vector<8x16xf32>
//       CHECK: vector.transfer_read %{{.*}} : memref<16x32xf32>, vector<16x32xf32>
//       CHECK: vector.transfer_read %{{.*}} : memref<8x32xf32>, vector<8x32xf32>
//       CHECK: vector.contract {indexing_maps = [#[[$mk]], #[[$kn]], #[[$mn]]], iterator_types = ["parallel", "parallel", "reduction"]} %{{.*}}, %{{.*}}, %{{.*}} : vector<8x16xf32>, vector<16x32xf32> into vector<8x32xf32>
//       CHECK: vector.transfer_write %{{.*}}, %{{.*}} : vector<8x32xf32>, memref<8x32xf32>

func @vectorization_test_integer(%A: memref<8x16xi32>, %B: memref<16x32xi32>,
                                 %C: memref<8x32xi32>) {
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
// CHECK-LABEL: func @vectorization_test_integer
//       CHECK: vector.transfer_read %{{.*}} : memref<8x16xi32>, vector<8x16xi32>
//       CHECK: vector.transfer_read %{{.*}} : memref<16x32xi32>, vector<16x32xi32>
//       CHECK: vector.transfer_read %{{.*}} : memref<8x32xi32>, vector<8x32xi32>
//       CHECK: vector.contract {indexing_maps = [#[[$mk]], #[[$kn]], #[[$mn]]], iterator_types = ["parallel", "parallel", "reduction"]} %{{.*}}, %{{.*}}, %{{.*}} : vector<8x16xi32>, vector<16x32xi32> into vector<8x32xi32>
//       CHECK: vector.transfer_write %{{.*}}, %{{.*}} : vector<8x32xi32>, memref<8x32xi32>

func @vectorization_test_2(%A: memref<8x16xf32>, %B: memref<16x32xf32>,
                         %C: memref<8x32xf32>) {
  linalg.matmul
    ins(%A, %B: memref<8x16xf32>, memref<16x32xf32>)
   outs(%C: memref<8x32xf32>)
  return
}
// CHECK-LABEL: func @vectorization_test_2
//       CHECK: vector.contract {{.*}} :
//                vector<8x16xf32>, vector<16x32xf32> into vector<8x32xf32>

func @test_vectorize_fill(%A : memref<8x16xf32>, %arg0 : f32) {
  linalg.fill(%A, %arg0) :  memref<8x16xf32>, f32
  return
}
// CHECK-LABEL: func @test_vectorize_fill
//       CHECK: %[[V:.*]] = vector.broadcast {{.*}} : f32 to vector<8x16xf32>
//       CHECK: vector.transfer_write %[[V]], {{.*}} : vector<8x16xf32>, memref<8x16xf32>

func @test_vectorize_fill_scalar(%A : memref<f32>, %arg0 : f32) {
  linalg.fill(%A, %arg0) :  memref<f32>, f32
  return
}
// CHECK-LABEL: func @test_vectorize_fill
//  CHECK-SAME: (%[[M:.*]]: memref<f32>, %[[V:.*]]: f32)
//       CHECK:   store %[[V]], %[[M]][] : memref<f32>

func @test_vectorize_copy(%A : memref<8x16xf32>, %B : memref<8x16xf32>) {
  linalg.copy(%A, %B) :  memref<8x16xf32>, memref<8x16xf32>
  return
}
// CHECK-LABEL: func @test_vectorize_copy
//       CHECK: %[[V:.*]] = vector.transfer_read {{.*}} : memref<8x16xf32>, vector<8x16xf32>
//       CHECK: vector.transfer_write %[[V]], {{.*}} : vector<8x16xf32>, memref<8x16xf32>

func @test_vectorize_copy_scalar(%A : memref<f32>, %B : memref<f32>) {
  linalg.copy(%A, %B) :  memref<f32>, memref<f32>
  return
}
// CHECK-LABEL: func @test_vectorize_copy_scalar
//       CHECK: %[[V:.*]] = load {{.*}} : memref<f32>
//       CHECK: store %[[V]], {{.*}} : memref<f32>

func @generic_vectorize(%arg0: memref<4x256xf32>, %arg1: memref<4x256xf32>,
  %arg2: memref<256xf32>, %i: f32) {
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
    %arg9 : f32, %arg10 : f32, %arg11 : f32, %arg12 : f32, %arg13 : f32,
    %arg14 : f32):
    %6 = addf %arg4, %arg6 : f32
    %7 = cmpf "ogt", %arg3, %arg6 : f32
    %8 = constant 2.0 : f32
    %9 = divf %arg5, %i : f32
    %10 = exp2 %arg5 : f32
    %11 = mulf %arg5, %8 : f32
    %12 = rsqrt %arg5 : f32
    %13 = select %7, %arg5, %arg6 : f32
    %14 = subf %arg5, %arg6 : f32
    %15 = tanh %arg5 : f32
    linalg.yield %6, %8, %c1_f32, %9, %10, %11, %12, %13, %14, %15 : f32, f32,
      f32, f32, f32, f32, f32, f32, f32, f32
  }
  return
}

// CHECK-LABEL: func @generic_vectorize
//  CHECK-SAME: (%[[ARG0:.*]]: memref<4x256xf32>, %[[ARG1:.*]]: memref<4x256xf32>,
//  CHECK-SAME:  %[[ARG2:.*]]: memref<256xf32>, %[[ARG3:.*]]: f32)
//   CHECK-DAG:   %[[CST0:.*]] = constant dense<2.000000e+00> : vector<4x256xf32>
//   CHECK-DAG:   %[[CST1:.*]] = constant dense<1.000000e+00> : vector<4x256xf32>
//   CHECK-DAG:   %[[C0:.*]] = constant 0 : index
//       CHECK:   %[[V0:.*]] = vector.transfer_read %[[ARG2]][%[[C0]]], {{.*}} : memref<256xf32>, vector<256xf32>
//       CHECK:   %[[V1:.*]] = vector.transfer_read %[[ARG0]][%[[C0]], %[[C0]]], {{.*}} : memref<4x256xf32>, vector<4x256xf32>
//       CHECK:   %[[V0B:.*]] = vector.broadcast %[[V0]] : vector<256xf32> to vector<4x256xf32>
//       CHECK:   %[[ADD:.*]] = addf %[[V0B]], %[[V1]] : vector<4x256xf32>
//       CHECK:   %[[V2:.*]] = vector.transfer_read %[[ARG1]][%[[C0]], %[[C0]]], {{.*}} : memref<4x256xf32>, vector<4x256xf32>
//       CHECK:   %[[CMP:.*]] = cmpf "ogt", %[[V2]], %[[V1]] : vector<4x256xf32>
//       CHECK:   %[[V3:.*]] = vector.transfer_read %[[ARG0]][%[[C0]], %[[C0]]], {{.*}} : memref<4x256xf32>, vector<4x256xf32> 
//       CHECK:   %[[ARG3B:.*]] = vector.broadcast %[[ARG3]] : f32 to vector<4x256xf32>
//       CHECK:   %[[DIV:.*]] = divf %[[V3]], %[[ARG3B]] : vector<4x256xf32>
//       CHECK:   %[[EXP:.*]] = exp2 %[[V3]] : vector<4x256xf32>
//       CHECK:   %[[MUL:.*]] = mulf %[[V3]], %[[CST0]] : vector<4x256xf32>
//       CHECK:   %[[RSQRT:.*]] = rsqrt %[[V3]] : vector<4x256xf32>
//       CHECK:   %[[SEL:.*]] = select %[[CMP]], %[[V3]], %[[V1]] : vector<4x256xi1>, vector<4x256xf32>
//       CHECK:   %[[SUB:.*]] = subf %[[V3]], %[[V1]] : vector<4x256xf32>
//       CHECK:   %[[TAN:.*]] = tanh %[[V3]] : vector<4x256xf32>
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
