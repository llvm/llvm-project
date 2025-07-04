// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(convert-vector-to-scf{full-unroll=true}))" -split-input-file -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func @transfer_read_inbounds
func.func @transfer_read_inbounds(%A : memref<?x?x?xf32>) -> (vector<2x3x4xf32>) {
  %f0 = arith.constant 0.0: f32
  %c0 = arith.constant 0: index

  // CHECK:      vector.transfer_read {{.*}} : memref<?x?x?xf32>, vector<4xf32>
  // CHECK-NEXT: vector.insert {{.*}} [0, 0] : vector<4xf32> into vector<2x3x4xf32>
  // CHECK-NEXT: vector.transfer_read {{.*}} : memref<?x?x?xf32>, vector<4xf32>
  // CHECK-NEXT: vector.insert {{.*}} [0, 1] : vector<4xf32> into vector<2x3x4xf32>
  // CHECK-NEXT: vector.transfer_read {{.*}} : memref<?x?x?xf32>, vector<4xf32>
  // CHECK-NEXT: vector.insert {{.*}} [0, 2] : vector<4xf32> into vector<2x3x4xf32>
  // CHECK-NEXT: vector.transfer_read {{.*}} : memref<?x?x?xf32>, vector<4xf32>
  // CHECK-NEXT: vector.insert {{.*}} [1, 0] : vector<4xf32> into vector<2x3x4xf32>
  // CHECK-NEXT: vector.transfer_read {{.*}} : memref<?x?x?xf32>, vector<4xf32>
  // CHECK-NEXT: vector.insert {{.*}} [1, 1] : vector<4xf32> into vector<2x3x4xf32>
  // CHECK-NEXT: vector.transfer_read {{.*}} : memref<?x?x?xf32>, vector<4xf32>
  // CHECK-NEXT: vector.insert {{.*}} [1, 2] : vector<4xf32> into vector<2x3x4xf32>
  // CHECK-NOT: scf.if
  // CHECK-NOT: scf.for
  %vec = vector.transfer_read %A[%c0, %c0, %c0], %f0 {in_bounds = [true, true, true]} : memref<?x?x?xf32>, vector<2x3x4xf32>
  return %vec : vector<2x3x4xf32>
}

// -----

// CHECK-LABEL: func @transfer_read_out_of_bounds
func.func @transfer_read_out_of_bounds(%A : memref<?x?x?xf32>) -> (vector<2x3x4xf32>) {
  %f0 = arith.constant 0.0: f32
  %c0 = arith.constant 0: index

  // CHECK: scf.if
  // CHECK: scf.if
  // CHECK: vector.transfer_read {{.*}} : memref<?x?x?xf32>, vector<4xf32>
  // CHECK: vector.insert {{.*}} [0, 0] : vector<4xf32> into vector<2x3x4xf32>
  // CHECK: scf.if
  // CHECK: vector.transfer_read {{.*}} : memref<?x?x?xf32>, vector<4xf32>
  // CHECK: vector.insert {{.*}} [0, 1] : vector<4xf32> into vector<2x3x4xf32>
  // CHECK: scf.if
  // CHECK: vector.transfer_read {{.*}} : memref<?x?x?xf32>, vector<4xf32>
  // CHECK: vector.insert {{.*}} [0, 2] : vector<4xf32> into vector<2x3x4xf32>
  // CHECK: scf.if
  // CHECK: scf.if
  // CHECK: vector.transfer_read {{.*}} : memref<?x?x?xf32>, vector<4xf32>
  // CHECK: vector.insert {{.*}} [1, 0] : vector<4xf32> into vector<2x3x4xf32>
  // CHECK: scf.if
  // CHECK: vector.transfer_read {{.*}} : memref<?x?x?xf32>, vector<4xf32>
  // CHECK: vector.insert {{.*}} [1, 1] : vector<4xf32> into vector<2x3x4xf32>
  // CHECK: scf.if
  // CHECK: vector.transfer_read {{.*}} : memref<?x?x?xf32>, vector<4xf32>
  // CHECK: vector.insert {{.*}} [1, 2] : vector<4xf32> into vector<2x3x4xf32>
  // CHECK-NOT: scf.for
  %vec = vector.transfer_read %A[%c0, %c0, %c0], %f0 : memref<?x?x?xf32>, vector<2x3x4xf32>
  return %vec : vector<2x3x4xf32>
}

// -----

func.func @transfer_read_mask(%A : memref<?x?x?xf32>, %mask : vector<2x3x4xi1>) -> (vector<2x3x4xf32>) {
  %f0 = arith.constant 0.0: f32
  %c0 = arith.constant 0: index

  // CHECK:      vector.extract %{{.*}}[0, 0] : vector<4xi1> from vector<2x3x4xi1>
  // CHECK-NEXT: vector.transfer_read {{.*}} : memref<?x?x?xf32>, vector<4xf32>
  // CHECK-NEXT: vector.insert {{.*}} [0, 0] : vector<4xf32> into vector<2x3x4xf32>
  // CHECK-NEXT: vector.extract %{{.*}}[0, 1] : vector<4xi1> from vector<2x3x4xi1>
  // CHECK-NEXT: vector.transfer_read {{.*}} : memref<?x?x?xf32>, vector<4xf32>
  // CHECK-NEXT: vector.insert {{.*}} [0, 1] : vector<4xf32> into vector<2x3x4xf32>
  // CHECK-NEXT: vector.extract %{{.*}}[0, 2] : vector<4xi1> from vector<2x3x4xi1>
  // CHECK-NEXT: vector.transfer_read {{.*}} : memref<?x?x?xf32>, vector<4xf32>
  // CHECK-NEXT: vector.insert {{.*}} [0, 2] : vector<4xf32> into vector<2x3x4xf32>
  // CHECK-NEXT: vector.extract %{{.*}}[1, 0] : vector<4xi1> from vector<2x3x4xi1>
  // CHECK-NEXT: vector.transfer_read {{.*}} : memref<?x?x?xf32>, vector<4xf32>
  // CHECK-NEXT: vector.insert {{.*}} [1, 0] : vector<4xf32> into vector<2x3x4xf32>
  // CHECK-NEXT: vector.extract %{{.*}}[1, 1] : vector<4xi1> from vector<2x3x4xi1>
  // CHECK-NEXT: vector.transfer_read {{.*}} : memref<?x?x?xf32>, vector<4xf32>
  // CHECK-NEXT: vector.insert {{.*}} [1, 1] : vector<4xf32> into vector<2x3x4xf32>
  // CHECK-NEXT: vector.extract %{{.*}}[1, 2] : vector<4xi1> from vector<2x3x4xi1>
  // CHECK-NEXT: vector.transfer_read {{.*}} : memref<?x?x?xf32>, vector<4xf32>
  // CHECK-NEXT: vector.insert {{.*}} [1, 2] : vector<4xf32> into vector<2x3x4xf32>
  // CHECK-NOT: scf.if
  // CHECK-NOT: scf.for
  %vec = vector.transfer_read %A[%c0, %c0, %c0], %f0, %mask {in_bounds = [true, true, true]}: memref<?x?x?xf32>, vector<2x3x4xf32>
  return %vec : vector<2x3x4xf32>
}

// -----

func.func @transfer_read_perm_mask(%A : memref<?x?x?x?xf32>, %mask : vector<3x2x4xi1>) -> (vector<2x3x4xf32>) {
  %f0 = arith.constant 0.0: f32
  %c0 = arith.constant 0: index

  // CHECK:      vector.extract %{{.*}}[0, 0] : vector<4xi1> from vector<3x2x4xi1>
  // CHECK-NEXT: vector.transfer_read {{.*}} : memref<?x?x?x?xf32>, vector<4xf32>
  // CHECK-NEXT: vector.insert {{.*}} [0, 0] : vector<4xf32> into vector<2x3x4xf32>
  // CHECK-NEXT: vector.extract %{{.*}}[1, 0] : vector<4xi1> from vector<3x2x4xi1>
  // CHECK-NEXT: vector.transfer_read {{.*}} : memref<?x?x?x?xf32>, vector<4xf32>
  // CHECK-NEXT: vector.insert {{.*}} [0, 1] : vector<4xf32> into vector<2x3x4xf32>
  // CHECK-NEXT: vector.extract %{{.*}}[2, 0] : vector<4xi1> from vector<3x2x4xi1>
  // CHECK-NEXT: vector.transfer_read {{.*}} : memref<?x?x?x?xf32>, vector<4xf32>
  // CHECK-NEXT: vector.insert {{.*}} [0, 2] : vector<4xf32> into vector<2x3x4xf32>
  // CHECK-NEXT: vector.extract %{{.*}}[0, 1] : vector<4xi1> from vector<3x2x4xi1>
  // CHECK-NEXT: vector.transfer_read {{.*}} : memref<?x?x?x?xf32>, vector<4xf32>
  // CHECK-NEXT: vector.insert {{.*}} [1, 0] : vector<4xf32> into vector<2x3x4xf32>
  // CHECK-NEXT: vector.extract %{{.*}}[1, 1] : vector<4xi1> from vector<3x2x4xi1>
  // CHECK-NEXT: vector.transfer_read {{.*}} : memref<?x?x?x?xf32>, vector<4xf32>
  // CHECK-NEXT: vector.insert {{.*}} [1, 1] : vector<4xf32> into vector<2x3x4xf32>
  // CHECK-NEXT: vector.extract %{{.*}}[2, 1] : vector<4xi1> from vector<3x2x4xi1>
  // CHECK-NEXT: vector.transfer_read {{.*}} : memref<?x?x?x?xf32>, vector<4xf32>
  // CHECK-NEXT: vector.insert {{.*}} [1, 2] : vector<4xf32> into vector<2x3x4xf32>
  // CHECK-NOT: scf.if
  // CHECK-NOT: scf.for
  %vec = vector.transfer_read %A[%c0, %c0, %c0, %c0], %f0, %mask {permutation_map = affine_map<(d0, d1, d2, d4) -> (d2, d0, d4)>, in_bounds = [true, true, true]}: memref<?x?x?x?xf32>, vector<2x3x4xf32>
  return %vec : vector<2x3x4xf32>
}
