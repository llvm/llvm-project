// RUN: mlir-opt %s -affine-super-vectorize="virtual-vector-size=8 test-fastest-varying=0" -split-input-file | FileCheck %s

// CHECK-DAG: #[[$MAP_ID0:map[0-9a-zA-Z_]*]] = affine_map<(d0) -> (d0 mod 12)>
// CHECK-DAG: #[[$MAP_ID1:map[0-9a-zA-Z_]*]] = affine_map<(d0) -> (d0 mod 16)>

// CHECK-LABEL: vec_affine_apply
// CHECK-SAME:  (%[[ARG0:.*]]: memref<8x12x16xf32>, %[[ARG1:.*]]: memref<8x24x48xf32>) {
func.func @vec_affine_apply(%arg0: memref<8x12x16xf32>, %arg1: memref<8x24x48xf32>) {
// CHECK:       affine.for %[[ARG2:.*]] = 0 to 8 {
// CHECK-NEXT:    affine.for %[[ARG3:.*]] = 0 to 24 {
// CHECK-NEXT:      affine.for %[[ARG4:.*]] = 0 to 48 step 8 {
// CHECK-NEXT:        %[[S0:.*]] = affine.apply #[[$MAP_ID0]](%[[ARG3]])
// CHECK-NEXT:        %[[S1:.*]] = affine.apply #[[$MAP_ID1]](%[[ARG4]])
// CHECK-NEXT:        %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:        %[[S2:.*]] = vector.transfer_read %[[ARG0]][%[[ARG2]], %[[S0]], %[[S1]]], %[[CST]] : memref<8x12x16xf32>, vector<8xf32>
// CHECK-NEXT:        vector.transfer_write %[[S2]], %[[ARG1]][%[[ARG2]], %[[ARG3]], %[[ARG4]]] : vector<8xf32>, memref<8x24x48xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  return
  affine.for %arg2 = 0 to 8 {
    affine.for %arg3 = 0 to 24 {
      affine.for %arg4 = 0 to 48 {
        %0 = affine.apply affine_map<(d0) -> (d0 mod 12)>(%arg3)
        %1 = affine.apply affine_map<(d0) -> (d0 mod 16)>(%arg4)
        %2 = affine.load %arg0[%arg2, %0, %1] : memref<8x12x16xf32>
        affine.store %2, %arg1[%arg2, %arg3, %arg4] : memref<8x24x48xf32>
      }
    }
  }
  return
}

// -----

// CHECK-DAG: #[[$MAP_ID2:map[0-9a-zA-Z_]*]] = affine_map<(d0) -> (d0 mod 16 + 1)>

// CHECK-LABEL: vec_affine_apply_2
// CHECK-SAME:  (%[[ARG0:.*]]: memref<8x12x16xf32>, %[[ARG1:.*]]: memref<8x24x48xf32>) {
func.func @vec_affine_apply_2(%arg0: memref<8x12x16xf32>, %arg1: memref<8x24x48xf32>) {
// CHECK:      affine.for %[[ARG2:.*]] = 0 to 8 {
// CHECK-NEXT:   affine.for %[[ARG3:.*]] = 0 to 12 {
// CHECK-NEXT:     affine.for %[[ARG4:.*]] = 0 to 48 step 8 {
// CHECK-NEXT:       %[[S0:.*]] = affine.apply #[[$MAP_ID2]](%[[ARG4]])
// CHECK-NEXT:       %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:       %[[S1:.*]] = vector.transfer_read %[[ARG0]][%[[ARG2]], %[[ARG3]], %[[S0]]], %[[CST]] : memref<8x12x16xf32>, vector<8xf32>
// CHECK-NEXT:       vector.transfer_write %[[S1]], %[[ARG1]][%[[ARG2]], %[[ARG3]], %[[ARG4]]] : vector<8xf32>, memref<8x24x48xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
  affine.for %arg2 = 0 to 8 {
    affine.for %arg3 = 0 to 12 {
      affine.for %arg4 = 0 to 48 {
        %1 = affine.apply affine_map<(d0) -> (d0 mod 16 + 1)>(%arg4)
        %2 = affine.load %arg0[%arg2, %arg3, %1] : memref<8x12x16xf32>
        affine.store %2, %arg1[%arg2, %arg3, %arg4] : memref<8x24x48xf32>
      }
    }
  }
  return
}

// -----

// CHECK-LABEL: no_vec_affine_apply
// CHECK-SAME:  (%[[ARG0:.*]]: memref<8x12x16xi32>, %[[ARG1:.*]]: memref<8x24x48xi32>) {
func.func @no_vec_affine_apply(%arg0: memref<8x12x16xi32>, %arg1: memref<8x24x48xi32>) {
// CHECK:      affine.for %[[ARG2:.*]] = 0 to 8 {
// CHECK-NEXT:   affine.for %[[ARG3:.*]] = 0 to 24 {
// CHECK-NEXT:     affine.for %[[ARG4:.*]] = 0 to 48 {
// CHECK-NEXT:       %[[S0:.*]] = affine.apply #[[$MAP_ID0]](%[[ARG3]])
// CHECK-NEXT:       %[[S1:.*]] = affine.apply #[[$MAP_ID1]](%[[ARG4]])
// CHECK-NEXT:       %[[S2:.*]] = affine.load %[[ARG0]][%[[ARG2]], %[[S0]], %[[S1]]] : memref<8x12x16xi32>
// CHECK-NEXT:       %[[S3:.*]] = arith.index_cast %[[S2]] : i32 to index
// CHECK-NEXT:       %[[S4:.*]] = affine.apply #[[$MAP_ID1]](%[[S3]])
// CHECK-NEXT:       %[[S5:.*]] = arith.index_cast %[[S4]] : index to i32
// CHECK-NEXT:       affine.store %[[S5]], %[[ARG1]][%[[ARG2]], %[[ARG3]], %[[ARG4]]] : memref<8x24x48xi32>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: return
  affine.for %arg2 = 0 to 8 {
    affine.for %arg3 = 0 to 24 {
      affine.for %arg4 = 0 to 48 {
        %0 = affine.apply affine_map<(d0) -> (d0 mod 12)>(%arg3)
        %1 = affine.apply affine_map<(d0) -> (d0 mod 16)>(%arg4)
        %2 = affine.load %arg0[%arg2, %0, %1] : memref<8x12x16xi32>
        %3 = arith.index_cast %2 : i32 to index
        %4 = affine.apply affine_map<(d0) -> (d0 mod 16)>(%3)
        %5 = arith.index_cast %4 : index to i32
        affine.store %5, %arg1[%arg2, %arg3, %arg4] : memref<8x24x48xi32>
      }
    }
  }
  return
}

// -----

// CHECK-DAG: #[[$MAP_ID1:map[0-9a-zA-Z_]*]] = affine_map<(d0) -> (d0 mod 16)>

// CHECK-LABEL: affine_map_with_expr
// CHECK-SAME:  (%[[ARG0:.*]]: memref<8x12x16xf32>, %[[ARG1:.*]]: memref<8x24x48xf32>) {
func.func @affine_map_with_expr(%arg0: memref<8x12x16xf32>, %arg1: memref<8x24x48xf32>) {
// CHECK:      affine.for %[[ARG2:.*]] = 0 to 8 {
// CHECK-NEXT:   affine.for %[[ARG3:.*]] = 0 to 12 {
// CHECK-NEXT:     affine.for %[[ARG4:.*]] = 0 to 48 {
// CHECK-NEXT:       %[[S0:.*]] = affine.apply #[[$MAP_ID1]](%[[ARG4]])
// CHECK-NEXT:       %[[S1:.*]] = affine.load %[[ARG0]][%[[ARG2]], %[[ARG3]], %[[S0]] + 1] : memref<8x12x16xf32>
// CHECK-NEXT:       affine.store %[[S1]], %[[ARG1]][%[[ARG2]], %[[ARG3]], %[[ARG4]]] : memref<8x24x48xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: return
  affine.for %arg2 = 0 to 8 {
    affine.for %arg3 = 0 to 12 {
      affine.for %arg4 = 0 to 48 {
        %1 = affine.apply affine_map<(d0) -> (d0 mod 16)>(%arg4)
        %2 = affine.load %arg0[%arg2, %arg3, %1 + 1] : memref<8x12x16xf32>
        affine.store %2, %arg1[%arg2, %arg3, %arg4] : memref<8x24x48xf32>
      }
    }
  }
  return
}

// -----

// CHECK-DAG: #[[$MAP_ID3:map[0-9a-zA-Z_]*]] = affine_map<(d0, d1, d2) -> (d0)>
// CHECK-DAG: #[[$MAP_ID4:map[0-9a-zA-Z_]*]] = affine_map<(d0, d1, d2) -> (d1)>
// CHECK-DAG: #[[$MAP_ID5:map[0-9a-zA-Z_]*]] = affine_map<(d0, d1, d2) -> (d2 + 1)>
// CHECK-DAG: #[[$MAP_ID6:map[0-9a-zA-Z_]*]] = affine_map<(d0, d1, d2) -> (0)>

// CHECK-LABEL: affine_map_with_expr_2
// CHECK-SAME:  (%[[ARG0:.*]]: memref<8x12x16xf32>, %[[ARG1:.*]]: memref<8x24x48xf32>, %[[I0:.*]]: index) {
func.func @affine_map_with_expr_2(%arg0: memref<8x12x16xf32>, %arg1: memref<8x24x48xf32>, %i: index) {
// CHECK:      affine.for %[[ARG3:.*]] = 0 to 8 {
// CHECK-NEXT:   affine.for %[[ARG4:.*]] = 0 to 12 {
// CHECK-NEXT:     affine.for %[[ARG5:.*]] = 0 to 48 step 8 {
// CHECK-NEXT:       %[[S0:.*]] = affine.apply #[[$MAP_ID3]](%[[ARG3]], %[[ARG4]], %[[I0]])
// CHECK-NEXT:       %[[S1:.*]] = affine.apply #[[$MAP_ID4]](%[[ARG3]], %[[ARG4]], %[[I0]])
// CHECK-NEXT:       %[[S2:.*]] = affine.apply #[[$MAP_ID5]](%[[ARG3]], %[[ARG4]], %[[I0]])
// CHECK-NEXT:       %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:       %[[S3:.*]] = vector.transfer_read %[[ARG0]][%[[S0]], %[[S1]], %[[S2]]], %[[CST]] {permutation_map = #[[$MAP_ID6]]} : memref<8x12x16xf32>, vector<8xf32>
// CHECK-NEXT:       vector.transfer_write %[[S3]], %[[ARG1]][%[[ARG3]], %[[ARG4]], %[[ARG5]]] : vector<8xf32>, memref<8x24x48xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: return
  affine.for %arg2 = 0 to 8 {
    affine.for %arg3 = 0 to 12 {
      affine.for %arg4 = 0 to 48 {
        %2 = affine.load %arg0[%arg2, %arg3, %i + 1] : memref<8x12x16xf32>
        affine.store %2, %arg1[%arg2, %arg3, %arg4] : memref<8x24x48xf32>
      }
    }
  }
  return
}
