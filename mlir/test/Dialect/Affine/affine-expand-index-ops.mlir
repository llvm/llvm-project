// RUN: mlir-opt %s -affine-expand-index-ops -split-input-file | FileCheck %s

// CHECK-LABEL: @delinearize_static_basis
//  CHECK-SAME:    (%[[IDX:.+]]: index)
//   CHECK-DAG:   %[[C224:.+]] = arith.constant 224 : index
//   CHECK-DAG:   %[[C50176:.+]] = arith.constant 50176 : index
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   %[[N:.+]] = arith.floordivsi %[[IDX]], %[[C50176]]
//   CHECK-DAG:   %[[P_REM:.+]] = arith.remsi %[[IDX]], %[[C50176]]
//   CHECK-DAG:   %[[P_NEG:.+]] = arith.cmpi slt, %[[P_REM]], %[[C0]]
//   CHECK-DAG:   %[[P_SHIFTED:.+]] = arith.addi %[[P_REM]], %[[C50176]]
//   CHECK-DAG:   %[[P_MOD:.+]] = arith.select %[[P_NEG]], %[[P_SHIFTED]], %[[P_REM]]
//       CHECK:   %[[P:.+]] = arith.divsi %[[P_MOD]], %[[C224]]
//   CHECK-DAG:   %[[Q_REM:.+]] = arith.remsi %[[IDX]], %[[C224]]
//   CHECK-DAG:   %[[Q_NEG:.+]] = arith.cmpi slt, %[[Q_REM]], %[[C0]]
//   CHECK-DAG:   %[[Q_SHIFTED:.+]] = arith.addi %[[Q_REM]], %[[C224]]
//       CHECK:   %[[Q:.+]] = arith.select %[[Q_NEG]], %[[Q_SHIFTED]], %[[Q_REM]]
//       CHECK:   return %[[N]], %[[P]], %[[Q]]
func.func @delinearize_static_basis(%linear_index: index) -> (index, index, index) {
  %1:3 = affine.delinearize_index %linear_index into (16, 224, 224) : index, index, index
  return %1#0, %1#1, %1#2 : index, index, index
}

// -----

// CHECK-LABEL: @delinearize_dynamic_basis
//  CHECK-SAME:    (%[[IDX:.+]]: index, %[[MEMREF:.+]]: memref
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//       CHECK:  %[[DIM1:.+]] = memref.dim %[[MEMREF]], %[[C1]] :
//       CHECK:  %[[DIM2:.+]] = memref.dim %[[MEMREF]], %[[C2]] :
//       CHECK:  %[[STRIDE1:.+]] = arith.muli %[[DIM2]], %[[DIM1]]
//       CHECK:  %[[N:.+]] = arith.floordivsi %[[IDX]], %[[STRIDE1]]
//   CHECK-DAG:  %[[P_REM:.+]] = arith.remsi %[[IDX]], %[[STRIDE1]]
//   CHECK-DAG:  %[[P_NEG:.+]] = arith.cmpi slt, %[[P_REM]], %[[C0]]
//   CHECK-DAG:  %[[P_SHIFTED:.+]] = arith.addi %[[P_REM]], %[[STRIDE1]]
//   CHECK-DAG:  %[[P_MOD:.+]] = arith.select %[[P_NEG]], %[[P_SHIFTED]], %[[P_REM]]
//       CHECK:  %[[P:.+]] = arith.divsi %[[P_MOD]], %[[DIM2]]
//   CHECK-DAG:  %[[Q_REM:.+]] = arith.remsi %[[IDX]], %[[DIM2]]
//   CHECK-DAG:  %[[Q_NEG:.+]] = arith.cmpi slt, %[[Q_REM]], %[[C0]]
//   CHECK-DAG:  %[[Q_SHIFTED:.+]] = arith.addi %[[Q_REM]], %[[DIM2]]
//       CHECK:  %[[Q:.+]] = arith.select %[[Q_NEG]], %[[Q_SHIFTED]], %[[Q_REM]]
//       CHECK:   return %[[N]], %[[P]], %[[Q]]
func.func @delinearize_dynamic_basis(%linear_index: index, %src: memref<?x?x?xf32>) -> (index, index, index) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %b1 = memref.dim %src, %c1 : memref<?x?x?xf32>
  %b2 = memref.dim %src, %c2 : memref<?x?x?xf32>
  // Note: no outer bound.
  %1:3 = affine.delinearize_index %linear_index into (%b1, %b2) : index, index, index
  return %1#0, %1#1, %1#2 : index, index, index
}

// -----

// CHECK-LABEL: @linearize_static
// CHECK-SAME: (%[[arg0:.+]]: index, %[[arg1:.+]]: index, %[[arg2:.+]]: index)
// CHECK-DAG: %[[C5:.+]] = arith.constant 5 : index
// CHECK-DAG: %[[C15:.+]] = arith.constant 15 : index
// CHECK: %[[scaled_0:.+]] = arith.muli %[[arg0]], %[[C15]]
// CHECK: %[[scaled_1:.+]] = arith.muli %[[arg1]], %[[C5]]
// CHECK: %[[val_0:.+]] = arith.addi %[[scaled_0]], %[[scaled_1]]
// CHECK: %[[val_1:.+]] = arith.addi %[[val_0]], %[[arg2]]
// CHECK: return %[[val_1]]
func.func @linearize_static(%arg0: index, %arg1: index, %arg2: index) -> index {
  %0 = affine.linearize_index [%arg0, %arg1, %arg2] by (2, 3, 5) : index
  func.return %0 : index
}

// -----

// CHECK-LABEL: @linearize_dynamic
// CHECK-SAME: (%[[arg0:.+]]: index, %[[arg1:.+]]: index, %[[arg2:.+]]: index, %[[arg3:.+]]: index, %[[arg4:.+]]: index)
// CHECK: %[[stride_0:.+]] = arith.muli %[[arg4]], %[[arg3]]
// CHECK: %[[scaled_0:.+]] = arith.muli %[[arg0]], %[[stride_0]]
// CHECK: %[[scaled_1:.+]] = arith.muli %[[arg1]], %[[arg4]]
// CHECK: %[[val_0:.+]] = arith.addi %[[scaled_0]], %[[scaled_1]]
// CHECK: %[[val_1:.+]] = arith.addi %[[val_0]], %[[arg2]]
// CHECK: return %[[val_1]]
func.func @linearize_dynamic(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index) -> index {
  // Note: no outer bounds
  %0 = affine.linearize_index [%arg0, %arg1, %arg2] by (%arg3, %arg4) : index
  func.return %0 : index
}

// -----

// CHECK-LABEL: @linearize_sort_adds
// CHECK-SAME: (%[[arg0:.+]]: memref<?xi32>, %[[arg1:.+]]: index, %[[arg2:.+]]: index)
// CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
// CHECK: scf.for %[[arg3:.+]] = %{{.*}} to %[[arg2]] step %{{.*}} {
// CHECK: scf.for %[[arg4:.+]] = %{{.*}} to %[[C4]] step %{{.*}} {
// CHECK: %[[stride_0:.+]] = arith.muli %[[arg2]], %[[C4]]
// CHECK: %[[scaled_0:.+]] = arith.muli %[[arg1]], %[[stride_0]]
// CHECK: %[[scaled_1:.+]] = arith.muli %[[arg4]], %[[arg2]]
// Note: even though %arg3 has a lower stride, we add it first
// CHECK: %[[val_0_2:.+]] = arith.addi %[[scaled_0]], %[[arg3]]
// CHECK: %[[val_1:.+]] = arith.addi %[[val_0_2]], %[[scaled_1]]
// CHECK: memref.store %{{.*}}, %[[arg0]][%[[val_1]]]
func.func @linearize_sort_adds(%arg0: memref<?xi32>, %arg1: index, %arg2: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c0_i32 = arith.constant 0 : i32
  scf.for %arg3 = %c0 to %arg2 step %c1 {
    scf.for %arg4 = %c0 to %c4 step %c1 {
      %idx = affine.linearize_index disjoint [%arg1, %arg4, %arg3] by (4, %arg2) : index
      memref.store %c0_i32, %arg0[%idx] : memref<?xi32>
    }
  }
  return
}
