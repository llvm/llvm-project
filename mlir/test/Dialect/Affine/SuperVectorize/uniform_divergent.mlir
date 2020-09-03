// RUN: mlir-opt %s -affine-super-vectorize="virtual-vector-size=128" -split-input-file | FileCheck %s

// Specific tests to check vectorization of uniform/divergent values.

// CHECK-LABEL: @uniform_arg
// CHECK-SAME:  %[[in:.*]]: memref<512xf32>,
// CHECK-SAME:  %[[uniform:.*]]: f32
func @uniform_arg(%in : memref<512xf32>, %uniform : f32) {
  affine.for %i = 0 to 512 {
    %ld = affine.load %in[%i] : memref<512xf32>
    %add = addf %ld, %uniform : f32
  }
  return
}

// CHECK-NEXT: %[[bcast:.*]] = vector.broadcast %[[uniform]] : f32 to vector<128xf32>
// CHECK-NEXT: affine.for
// CHECK:        addf %{{.*}}, %[[bcast]] : vector<128xf32>

// -----

// CHECK-LABEL: @multi_use_uniform_arg
// CHECK-SAME:  %[[in:.*]]: memref<512xf32>
// CHECK-SAME:  %[[uniform:.*]]: f32
func @multi_use_uniform_arg(%in : memref<512xf32>, %uniform : f32) {
  affine.for %i = 0 to 512 {
    %ld = affine.load %in[%i] : memref<512xf32>
    %user0 = addf %ld, %uniform : f32
    %user1 = addf %ld, %uniform : f32
  }
  return
}

// CHECK-NEXT: %[[bcast:.*]] = vector.broadcast %[[uniform]] : f32 to vector<128xf32>
// CHECK-NOT:  vector.broadcast
// CHECK-NEXT: affine.for
// CHECK:        addf %{{.*}}, %[[bcast]] : vector<128xf32>
// CHECK:        addf %{{.*}}, %[[bcast]] : vector<128xf32>

// -----

// CHECK-LABEL: @uniform_load
func @uniform_load(%A : memref<?x?xf32>, %C : memref<?x?xf32>) {
  %c0 = constant 0 : index
  %N = dim %A, %c0 : memref<?x?xf32>
  affine.for %i = 0 to %N {
    %uniform_ld = affine.load %A[%i, %i] : memref<?x?xf32>
    affine.for %j = 0 to %N {
      %b = affine.load %A[%i, %j] : memref<?x?xf32>
      %c = addf %uniform_ld, %b : f32
    }
  }
  return
}

// CHECK:      affine.for
// CHECK-NEXT:   %[[uniform_ld:.*]] = affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32>
// CHECK-NEXT:   %[[bcast:.*]] = vector.broadcast %[[uniform_ld]] : f32 to vector<128xf32>
// CHECK-NEXT:   affine.for
// CHECK:          addf %[[bcast]], %{{.*}} : vector<128xf32>
