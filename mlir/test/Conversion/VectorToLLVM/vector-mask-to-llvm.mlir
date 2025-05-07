// RUN: mlir-opt %s --convert-vector-to-llvm='force-32bit-vector-indices=1' | FileCheck %s --check-prefix=CMP32
// RUN: mlir-opt %s --convert-vector-to-llvm='force-32bit-vector-indices=0' | FileCheck %s --check-prefix=CMP64

// CMP32-LABEL: @genbool_var_1d(
// CMP32-SAME: %[[ARG:.*]]: index)
// CMP32: %[[T0:.*]] = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]> : vector<11xi32>
// CMP32: %[[T1:.*]] = arith.index_cast %[[ARG]] : index to i32
// CMP32: %[[T2:.*]] = llvm.insertelement %[[T1]], %{{.*}}[%{{.*}} : i32] : vector<11xi32>
// CMP32: %[[T3:.*]] = llvm.shufflevector %[[T2]], %{{.*}} [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<11xi32>
// CMP32: %[[T4:.*]] = arith.cmpi sgt, %[[T3]], %[[T0]] : vector<11xi32>
// CMP32: return %[[T4]] : vector<11xi1>

// CMP64-LABEL: @genbool_var_1d(
// CMP64-SAME: %[[ARG:.*]]: index)
// CMP64: %[[T0:.*]] = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]> : vector<11xi64>
// CMP64: %[[T1:.*]] = arith.index_cast %[[ARG]] : index to i64
// CMP64: %[[T2:.*]] = llvm.insertelement %[[T1]], %{{.*}}[%{{.*}} : i32] : vector<11xi64>
// CMP64: %[[T3:.*]] = llvm.shufflevector %[[T2]], %{{.*}} [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<11xi64>
// CMP64: %[[T4:.*]] = arith.cmpi sgt, %[[T3]], %[[T0]] : vector<11xi64>
// CMP64: return %[[T4]] : vector<11xi1>

func.func @genbool_var_1d(%arg0: index) -> vector<11xi1> {
  %0 = vector.create_mask %arg0 : vector<11xi1>
  return %0 : vector<11xi1>
}

// CMP32-LABEL: @genbool_var_1d_scalable(
// CMP32-SAME: %[[ARG:.*]]: index)
// CMP32: %[[T0:.*]] = llvm.intr.stepvector : vector<[11]xi32>
// CMP32: %[[T1:.*]] = arith.index_cast %[[ARG]] : index to i32
// CMP32: %[[T2:.*]] = llvm.insertelement %[[T1]], %{{.*}}[%{{.*}} : i32] : vector<[11]xi32>
// CMP32: %[[T3:.*]] = llvm.shufflevector %[[T2]], %{{.*}} [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<[11]xi32>
// CMP32: %[[T4:.*]] = arith.cmpi slt, %[[T0]], %[[T3]] : vector<[11]xi32>
// CMP32: return %[[T4]] : vector<[11]xi1>

// CMP64-LABEL: @genbool_var_1d_scalable(
// CMP64-SAME: %[[ARG:.*]]: index)
// CMP64: %[[T0:.*]] = llvm.intr.stepvector : vector<[11]xi64>
// CMP64: %[[T1:.*]] = arith.index_cast %[[ARG]] : index to i64
// CMP64: %[[T2:.*]] = llvm.insertelement %[[T1]], %{{.*}}[%{{.*}} : i32] : vector<[11]xi64>
// CMP64: %[[T3:.*]] = llvm.shufflevector %[[T2]], %{{.*}} [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<[11]xi64>
// CMP64: %[[T4:.*]] = arith.cmpi slt, %[[T0]], %[[T3]] : vector<[11]xi64>
// CMP64: return %[[T4]] : vector<[11]xi1>

func.func @genbool_var_1d_scalable(%arg0: index) -> vector<[11]xi1> {
  %0 = vector.create_mask %arg0 : vector<[11]xi1>
  return %0 : vector<[11]xi1>
}
