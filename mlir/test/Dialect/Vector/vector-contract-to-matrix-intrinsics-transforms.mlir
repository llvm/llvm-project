// RUN: mlir-opt %s  --convert-vector-to-llvm='vector-contract-lowering=matmul' | FileCheck %s

#matmat_accesses = [
  affine_map<(i, j, k) -> (i, k)>,
  affine_map<(i, j, k) -> (k, j)>,
  affine_map<(i, j, k) -> (i, j)>
]
#matmat_trait = {
  indexing_maps = #matmat_accesses,
  iterator_types = ["parallel", "parallel", "reduction"]
}

// CHECK-LABEL:   func.func @matmul(
// CHECK-SAME:                      %[[ARG0:.*]]: vector<2x4xf32>,
// CHECK-SAME:                      %[[ARG1:.*]]: vector<4x3xf32>,
// CHECK-SAME:                      %[[ARG2:.*]]: vector<2x3xf32>) -> vector<2x3xf32> {
// CHECK:           %[[VAL_0:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : vector<4x3xf32> to !llvm.array<4 x vector<3xf32>>
// CHECK:           %[[VAL_1:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : vector<2x4xf32> to !llvm.array<2 x vector<4xf32>>
// CHECK:           %[[VAL_2:.*]] = ub.poison : vector<2x3xf32>
// CHECK:           %[[VAL_3:.*]] = builtin.unrealized_conversion_cast %[[VAL_2]] : vector<2x3xf32> to !llvm.array<2 x vector<3xf32>>
// CHECK:           %[[POISON_RHS:.*]] = ub.poison : vector<12xf32>
// CHECK:           %[[POISON_LHS:.*]] = ub.poison : vector<8xf32>

// ===> Extract LHS
//       | ROW_1 |
//       | ----- | --> | ROW_1 | ROW_2 |
//       | ROW_2 |
//
// CHECK:           %[[LHS_ROW_1:.*]] = llvm.extractvalue %[[VAL_1]][0] : !llvm.array<2 x vector<4xf32>>
// CHECK:           %[[TP_1:.*]] = llvm.shufflevector %[[LHS_ROW_1]], %[[LHS_ROW_1]] [0, 1, 2, 3, 0, 0, 0, 0] : vector<4xf32>
// CHECK:           %[[TP_2:.*]] = llvm.shufflevector %[[TP_1]], %[[POISON_LHS]] [0, 1, 2, 3, 12, 13, 14, 15] : vector<8xf32>
// CHECK:           %[[LHS_ROW_2:.*]] = llvm.extractvalue %[[VAL_1]][1] : !llvm.array<2 x vector<4xf32>>
// CHECK:           %[[TP_3:.*]] = llvm.shufflevector %[[LHS_ROW_2]], %[[LHS_ROW_2]] [0, 1, 2, 3, 0, 0, 0, 0] : vector<4xf32>
// CHECK:           %[[LHS:.*]] = llvm.shufflevector %[[TP_3]], %[[TP_2]] [8, 9, 10, 11, 0, 1, 2, 3] : vector<8xf32>

// == Extract RHS
//       | ROW_1 |
//       | ----- |
//       | ROW_2 |
//       | ----- | --> | ROW_1 | ROW_2 | ROW_3 | ROW_4 |
//       | ROW_3 |
//       | ----- |
//       | ROW_4 |
// CHECK:           %[[RHS_ROW_1:.*]] = llvm.extractvalue %[[VAL_0]][0] : !llvm.array<4 x vector<3xf32>>
// CHECK:           %[[TP_4:.*]] = llvm.shufflevector %[[RHS_ROW_1]], %[[RHS_ROW_1]] [0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<3xf32>
// CHECK:           %[[TP_5:.*]] = llvm.shufflevector %[[TP_4]], %[[POISON_RHS]] [0, 1, 2, 15, 16, 17, 18, 19, 20, 21, 22, 23] : vector<12xf32>
// CHECK:           %[[RHS_ROW_2:.*]] = llvm.extractvalue %[[VAL_0]][1] : !llvm.array<4 x vector<3xf32>>
// CHECK:           %[[TP_6:.*]] = llvm.shufflevector %[[RHS_ROW_2]], %[[RHS_ROW_2]] [0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<3xf32>
// CHECK:           %[[TP_7:.*]] = llvm.shufflevector %[[TP_6]], %[[TP_5]] [12, 13, 14, 0, 1, 2, 18, 19, 20, 21, 22, 23] : vector<12xf32>
// CHECK:           %[[RHS_ROW_3:.*]] = llvm.extractvalue %[[VAL_0]][2] : !llvm.array<4 x vector<3xf32>>
// CHECK:           %[[TP_8:.*]] = llvm.shufflevector %[[RHS_ROW_3]], %[[RHS_ROW_3]] [0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<3xf32>
// CHECK:           %[[TP_9:.*]] = llvm.shufflevector %[[TP_8]], %[[TP_7]] [12, 13, 14, 15, 16, 17, 0, 1, 2, 21, 22, 23] : vector<12xf32>
// CHECK:           %[[RHS_ROW_4:.*]] = llvm.extractvalue %[[VAL_0]][3] : !llvm.array<4 x vector<3xf32>>
// CHECK:           %[[TP_10:.*]] = llvm.shufflevector %[[RHS_ROW_4]], %[[RHS_ROW_4]] [0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<3xf32>
// CHECK:           %[[RHS:.*]] = llvm.shufflevector %[[TP_10]], %[[TP_9]] [12, 13, 14, 15, 16, 17, 18, 19, 20, 0, 1, 2] : vector<12xf32>

// ===> Matrix multiply
// CHECK:           %[[MM:.*]] = llvm.intr.matrix.multiply %[[LHS]], %[[RHS]] {lhs_columns = 4 : i32, lhs_rows = 2 : i32, rhs_columns = 3 : i32} : (vector<8xf32>, vector<12xf32>) -> vector<6xf32>
// CHECK:           %[[RES:.*]] = arith.addf %[[ARG2]], %{{.*}} : vector<2x3xf32>
// CHECK:           return %[[RES]] : vector<2x3xf32>
func.func @matmul(%arg0: vector<2x4xf32>,
                  %arg1: vector<4x3xf32>,
                  %arg2: vector<2x3xf32>) -> vector<2x3xf32> {
  %0 = vector.contract #matmat_trait %arg0, %arg1, %arg2
    : vector<2x4xf32>, vector<4x3xf32> into vector<2x3xf32>
  return %0 : vector<2x3xf32>
}

// CHECK-LABEL: func @matmul_scalable
// CHECK-NOT: llvm.intr.matrix.multiply
func.func @matmul_scalable(%arg0: vector<2x4xf32>,
                           %arg1: vector<4x[3]xf32>,
                           %arg2: vector<2x[3]xf32>) -> vector<2x[3]xf32> {
  %0 = vector.contract #matmat_trait %arg0, %arg1, %arg2
    : vector<2x4xf32>, vector<4x[3]xf32> into vector<2x[3]xf32>
  return %0 : vector<2x[3]xf32>
}
