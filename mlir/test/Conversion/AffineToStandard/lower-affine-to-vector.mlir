// RUN: mlir-opt -lower-affine --split-input-file %s | FileCheck %s

// CHECK: #[[perm_map:.*]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func @affine_vector_load
func @affine_vector_load(%arg0 : index) {
  %0 = alloc() : memref<100xf32>
  affine.for %i0 = 0 to 16 {
    %1 = affine.vector_load %0[%i0 + symbol(%arg0) + 7] : memref<100xf32>, vector<8xf32>
  }
// CHECK:       %[[buf:.*]] = alloc
// CHECK:       %[[a:.*]] = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:  %[[c7:.*]] = constant 7 : index
// CHECK-NEXT:  %[[b:.*]] = addi %[[a]], %[[c7]] : index
// CHECK-NEXT:  %[[pad:.*]] = constant 0.0
// CHECK-NEXT:  vector.transfer_read %[[buf]][%[[b]]], %[[pad]] {permutation_map = #[[perm_map]]} : memref<100xf32>, vector<8xf32>
  return
}

// -----

// CHECK: #[[perm_map:.*]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func @affine_vector_store
func @affine_vector_store(%arg0 : index) {
  %0 = alloc() : memref<100xf32>
  %1 = constant dense<11.0> : vector<4xf32>
  affine.for %i0 = 0 to 16 {
    affine.vector_store %1, %0[%i0 - symbol(%arg0) + 7] : memref<100xf32>, vector<4xf32>
}
// CHECK:       %[[buf:.*]] = alloc
// CHECK:       %[[val:.*]] = constant dense
// CHECK:       %[[c_1:.*]] = constant -1 : index
// CHECK-NEXT:  %[[a:.*]] = muli %arg0, %[[c_1]] : index
// CHECK-NEXT:  %[[b:.*]] = addi %{{.*}}, %[[a]] : index
// CHECK-NEXT:  %[[c7:.*]] = constant 7 : index
// CHECK-NEXT:  %[[c:.*]] = addi %[[b]], %[[c7]] : index
// CHECK-NEXT:  vector.transfer_write  %[[val]], %[[buf]][%[[c]]] {permutation_map = #[[perm_map]]} : vector<4xf32>, memref<100xf32>
  return
}

// -----

// CHECK: #[[perm_map:.*]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func @affine_vector_load
func @affine_vector_load(%arg0 : index) {
  %0 = alloc() : memref<100xf32>
  affine.for %i0 = 0 to 16 {
    %1 = affine.vector_load %0[%i0 + symbol(%arg0) + 7] : memref<100xf32>, vector<8xf32>
  }
// CHECK:       %[[buf:.*]] = alloc
// CHECK:       %[[a:.*]] = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:  %[[c7:.*]] = constant 7 : index
// CHECK-NEXT:  %[[b:.*]] = addi %[[a]], %[[c7]] : index
// CHECK-NEXT:  %[[pad:.*]] = constant 0.0
// CHECK-NEXT:  vector.transfer_read %[[buf]][%[[b]]], %[[pad]] {permutation_map = #[[perm_map]]} : memref<100xf32>, vector<8xf32>
  return
}

// -----

// CHECK: #[[perm_map:.*]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func @affine_vector_store
func @affine_vector_store(%arg0 : index) {
  %0 = alloc() : memref<100xf32>
  %1 = constant dense<11.0> : vector<4xf32>
  affine.for %i0 = 0 to 16 {
    affine.vector_store %1, %0[%i0 - symbol(%arg0) + 7] : memref<100xf32>, vector<4xf32>
}
// CHECK:       %[[buf:.*]] = alloc
// CHECK:       %[[val:.*]] = constant dense
// CHECK:       %[[c_1:.*]] = constant -1 : index
// CHECK-NEXT:  %[[a:.*]] = muli %arg0, %[[c_1]] : index
// CHECK-NEXT:  %[[b:.*]] = addi %{{.*}}, %[[a]] : index
// CHECK-NEXT:  %[[c7:.*]] = constant 7 : index
// CHECK-NEXT:  %[[c:.*]] = addi %[[b]], %[[c7]] : index
// CHECK-NEXT:  vector.transfer_write  %[[val]], %[[buf]][%[[c]]] {permutation_map = #[[perm_map]]} : vector<4xf32>, memref<100xf32>
  return
}

// -----

// CHECK: #[[perm_map:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @vector_load_2d
func @vector_load_2d() {
  %0 = alloc() : memref<100x100xf32>
  affine.for %i0 = 0 to 16 step 2{
    affine.for %i1 = 0 to 16 step 8 {
      %1 = affine.vector_load %0[%i0, %i1] : memref<100x100xf32>, vector<2x8xf32>
// CHECK:      %[[buf:.*]] = alloc
// CHECK:      scf.for %[[i0:.*]] =
// CHECK:        scf.for %[[i1:.*]] =
// CHECK-NEXT:     %[[pad:.*]] = constant 0.0
// CHECK-NEXT:     vector.transfer_read %[[buf]][%[[i0]], %[[i1]]], %[[pad]] {permutation_map = #[[perm_map]]} : memref<100x100xf32>, vector<2x8xf32>
    }
  }
  return
}

// -----

// CHECK: #[[perm_map:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @vector_store_2d
func @vector_store_2d() {
  %0 = alloc() : memref<100x100xf32>
  %1 = constant dense<11.0> : vector<2x8xf32>
  affine.for %i0 = 0 to 16 step 2{
    affine.for %i1 = 0 to 16 step 8 {
      affine.vector_store %1, %0[%i0, %i1] : memref<100x100xf32>, vector<2x8xf32>
// CHECK:      %[[buf:.*]] = alloc
// CHECK:      %[[val:.*]] = constant dense
// CHECK:      scf.for %[[i0:.*]] =
// CHECK:        scf.for %[[i1:.*]] =
// CHECK-NEXT:     vector.transfer_write  %[[val]], %[[buf]][%[[i0]], %[[i1]]] {permutation_map = #[[perm_map]]} : vector<2x8xf32>, memref<100x100xf32>
    }
  }
  return
}

