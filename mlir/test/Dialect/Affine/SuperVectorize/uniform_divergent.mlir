// RUN: mlir-opt %s -affine-super-vectorize="virtual-vector-size=128" -split-input-file | FileCheck %s

// Specific tests to check vectorization of uniform/divergent values.

// CHECK-LABEL: @uniform_arg
// CHECK-SAME:  %[[in:.*]]: memref<512xf32>,
// CHECK-SAME:  %[[uniform:.*]]: f32
func.func @uniform_arg(%in : memref<512xf32>, %uniform : f32) {
  affine.for %i = 0 to 512 {
    %ld = affine.load %in[%i] : memref<512xf32>
    %add = arith.addf %ld, %uniform : f32
  }
  return
}

// CHECK-NEXT: %[[bcast:.*]] = vector.broadcast %[[uniform]] : f32 to vector<128xf32>
// CHECK-NEXT: affine.for
// CHECK:        arith.addf %{{.*}}, %[[bcast]] : vector<128xf32>

// -----

// CHECK-LABEL: @multi_use_uniform_arg
// CHECK-SAME:  %[[in:.*]]: memref<512xf32>
// CHECK-SAME:  %[[uniform:.*]]: f32
func.func @multi_use_uniform_arg(%in : memref<512xf32>, %uniform : f32) {
  affine.for %i = 0 to 512 {
    %ld = affine.load %in[%i] : memref<512xf32>
    %user0 = arith.addf %ld, %uniform : f32
    %user1 = arith.addf %ld, %uniform : f32
  }
  return
}

// CHECK-NEXT: %[[bcast:.*]] = vector.broadcast %[[uniform]] : f32 to vector<128xf32>
// CHECK-NOT:  vector.broadcast
// CHECK-NEXT: affine.for
// CHECK:        arith.addf %{{.*}}, %[[bcast]] : vector<128xf32>
// CHECK:        arith.addf %{{.*}}, %[[bcast]] : vector<128xf32>

// -----

// CHECK-LABEL: @uniform_load
func.func @uniform_load(%A : memref<?x?xf32>, %C : memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %N = memref.dim %A, %c0 : memref<?x?xf32>
  affine.for %i = 0 to %N {
    %uniform_ld = affine.load %A[%i, %i] : memref<?x?xf32>
    affine.for %j = 0 to %N {
      %b = affine.load %A[%i, %j] : memref<?x?xf32>
      %c = arith.addf %uniform_ld, %b : f32
    }
  }
  return
}

// CHECK:      affine.for
// CHECK-NEXT:   %[[uniform_ld:.*]] = affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32>
// CHECK-NEXT:   %[[bcast:.*]] = vector.broadcast %[[uniform_ld]] : f32 to vector<128xf32>
// CHECK-NEXT:   affine.for
// CHECK:          arith.addf %[[bcast]], %{{.*}} : vector<128xf32>

// -----

// CHECK-LABEL: @uniform_multi_result
func.func @uniform_multi_result(%in : memref<512xi32>, %a : i32, %b : i32) {
  %sum, %overflow = arith.addui_extended %a, %b : i32, i1
  affine.for %i = 0 to 512 {
    %ld = affine.load %in[%i] : memref<512xi32>
    %add = arith.addi %ld, %sum : i32
  }
  return
}

// CHECK:      %[[sum:.*]], %{{.*}} = arith.addui_extended
// CHECK-NEXT: %[[bcast:.*]] = vector.broadcast %[[sum]] : i32 to vector<128xi32>
// CHECK-NEXT: affine.for
// CHECK:        arith.addi %{{.*}}, %[[bcast]] : vector<128xi32>
