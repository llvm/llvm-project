// RUN: mlir-opt %s --convert-vector-to-llvm='enable-index-optimizations=1' | FileCheck %s --check-prefix=CMP32
// RUN: mlir-opt %s --convert-vector-to-llvm='enable-index-optimizations=0' | FileCheck %s --check-prefix=CMP64

// CMP32-LABEL: llvm.func @genbool_var_1d(
// CMP32-SAME: %[[A:.*]]: !llvm.i64)
// CMP32: %[[T0:.*]] = llvm.mlir.constant(dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]> : vector<11xi32>) : !llvm.vec<11 x i32>
// CMP32: %[[T1:.*]] = llvm.trunc %[[A]] : !llvm.i64 to !llvm.i32
// CMP32: %[[T2:.*]] = llvm.mlir.undef : !llvm.vec<11 x i32>
// CMP32: %[[T3:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CMP32: %[[T4:.*]] = llvm.insertelement %[[T1]], %[[T2]][%[[T3]] : !llvm.i32] : !llvm.vec<11 x i32>
// CMP32: %[[T5:.*]] = llvm.shufflevector %[[T4]], %[[T2]] [0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32] : !llvm.vec<11 x i32>, !llvm.vec<11 x i32>
// CMP32: %[[T6:.*]] = llvm.icmp "slt" %[[T0]], %[[T5]] : !llvm.vec<11 x i32>
// CMP32: llvm.return %[[T6]] : !llvm.vec<11 x i1>

// CMP64-LABEL: llvm.func @genbool_var_1d(
// CMP64-SAME: %[[A:.*]]: !llvm.i64)
// CMP64: %[[T0:.*]] = llvm.mlir.constant(dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]> : vector<11xi64>) : !llvm.vec<11 x i64>
// CMP64: %[[T1:.*]] = llvm.mlir.undef : !llvm.vec<11 x i64>
// CMP64: %[[T2:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CMP64: %[[T3:.*]] = llvm.insertelement %[[A]], %[[T1]][%[[T2]] : !llvm.i32] : !llvm.vec<11 x i64>
// CMP64: %[[T4:.*]] = llvm.shufflevector %[[T3]], %[[T1]] [0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32] : !llvm.vec<11 x i64>, !llvm.vec<11 x i64>
// CMP64: %[[T5:.*]] = llvm.icmp "slt" %[[T0]], %[[T4]] : !llvm.vec<11 x i64>
// CMP64: llvm.return %[[T5]] : !llvm.vec<11 x i1>

func @genbool_var_1d(%arg0: index) -> vector<11xi1> {
  %0 = vector.create_mask %arg0 : vector<11xi1>
  return %0 : vector<11xi1>
}

// CMP32-LABEL: llvm.func @transfer_read_1d
// CMP32: %[[C:.*]] = llvm.mlir.constant(dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : vector<16xi32>) : !llvm.vec<16 x i32>
// CMP32: %[[A:.*]] = llvm.add %{{.*}}, %[[C]] : !llvm.vec<16 x i32>
// CMP32: %[[M:.*]] = llvm.icmp "slt" %[[A]], %{{.*}} : !llvm.vec<16 x i32>
// CMP32: %[[L:.*]] = llvm.intr.masked.load %{{.*}}, %[[M]], %{{.*}}
// CMP32: llvm.return %[[L]] : !llvm.vec<16 x float>

// CMP64-LABEL: llvm.func @transfer_read_1d
// CMP64: %[[C:.*]] = llvm.mlir.constant(dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : vector<16xi64>) : !llvm.vec<16 x i64>
// CMP64: %[[A:.*]] = llvm.add %{{.*}}, %[[C]] : !llvm.vec<16 x i64>
// CMP64: %[[M:.*]] = llvm.icmp "slt" %[[A]], %{{.*}} : !llvm.vec<16 x i64>
// CMP64: %[[L:.*]] = llvm.intr.masked.load %{{.*}}, %[[M]], %{{.*}}
// CMP64: llvm.return %[[L]] : !llvm.vec<16 x float>

func @transfer_read_1d(%A : memref<?xf32>, %i: index) -> vector<16xf32> {
  %d = constant -1.0: f32
  %f = vector.transfer_read %A[%i], %d {permutation_map = affine_map<(d0) -> (d0)>} : memref<?xf32>, vector<16xf32>
  return %f : vector<16xf32>
}
