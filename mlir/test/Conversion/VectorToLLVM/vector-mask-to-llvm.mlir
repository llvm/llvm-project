// RUN: mlir-opt %s --convert-vector-to-llvm='enable-index-optimizations=1' | FileCheck %s --check-prefix=CMP32
// RUN: mlir-opt %s --convert-vector-to-llvm='enable-index-optimizations=0' | FileCheck %s --check-prefix=CMP64

// CMP32-LABEL: @genbool_var_1d(
// CMP32-SAME: %[[ARG:.*]]: index)
// CMP32: %[[T0:.*]] = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]> : vector<11xi32>
// CMP32: %[[T1:.*]] = arith.index_cast %[[ARG]] : index to i32
// CMP32: %[[T2:.*]] = splat %[[T1]] : vector<11xi32>
// CMP32: %[[T3:.*]] = arith.cmpi slt, %[[T0]], %[[T2]] : vector<11xi32>
// CMP32: return %[[T3]] : vector<11xi1>

// CMP64-LABEL: @genbool_var_1d(
// CMP64-SAME: %[[ARG:.*]]: index)
// CMP64: %[[T0:.*]] = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]> : vector<11xi64>
// CMP64: %[[T1:.*]] = arith.index_cast %[[ARG]] : index to i64
// CMP64: %[[T2:.*]] = splat %[[T1]] : vector<11xi64>
// CMP64: %[[T3:.*]] = arith.cmpi slt, %[[T0]], %[[T2]] : vector<11xi64>
// CMP64: return %[[T3]] : vector<11xi1>

func @genbool_var_1d(%arg0: index) -> vector<11xi1> {
  %0 = vector.create_mask %arg0 : vector<11xi1>
  return %0 : vector<11xi1>
}

// CMP32-LABEL: @transfer_read_1d
// CMP32: %[[C:.*]] = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : vector<16xi32>
// CMP32: %[[A:.*]] = arith.addi %{{.*}}, %[[C]] : vector<16xi32>
// CMP32: %[[M:.*]] = arith.cmpi slt, %[[A]], %{{.*}} : vector<16xi32>
// CMP32: %[[L:.*]] = llvm.intr.masked.load %{{.*}}, %[[M]], %{{.*}}
// CMP32: return %[[L]] : vector<16xf32>

// CMP64-LABEL: @transfer_read_1d
// CMP64: %[[C:.*]] = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : vector<16xi64>
// CMP64: %[[A:.*]] = arith.addi %{{.*}}, %[[C]] : vector<16xi64>
// CMP64: %[[M:.*]] = arith.cmpi slt, %[[A]], %{{.*}} : vector<16xi64>
// CMP64: %[[L:.*]] = llvm.intr.masked.load %{{.*}}, %[[M]], %{{.*}}
// CMP64: return %[[L]] : vector<16xf32>

func @transfer_read_1d(%A : memref<?xf32>, %i: index) -> vector<16xf32> {
  %d = arith.constant -1.0: f32
  %f = vector.transfer_read %A[%i], %d {permutation_map = affine_map<(d0) -> (d0)>} : memref<?xf32>, vector<16xf32>
  return %f : vector<16xf32>
}
