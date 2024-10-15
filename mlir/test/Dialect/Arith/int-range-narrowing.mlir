// RUN: mlir-opt --arith-int-range-narrowing="target-bitwidth=32"  %s | FileCheck %s

// Do not truncate negative values
// CHECK-LABEL: func @test_addi_neg
//       CHECK:  %[[RES:.*]] = arith.addi %{{.*}}, %{{.*}} : index
//       CHECK:  return %[[RES]] : index
func.func @test_addi_neg() -> index {
  %0 = test.with_bounds { umin = 0 : index, umax = 1 : index, smin = 0 : index, smax = 1 : index } : index
  %1 = test.with_bounds { umin = 0 : index, umax = -1 : index, smin = -1 : index, smax = 0 : index } : index
  %2 = arith.addi %0, %1 : index
  return %2 : index
}

// CHECK-LABEL: func @test_addi
//       CHECK:  %[[A:.*]] = test.with_bounds {smax = 5 : index, smin = 4 : index, umax = 5 : index, umin = 4 : index} : index
//       CHECK:  %[[B:.*]] = test.with_bounds {smax = 7 : index, smin = 6 : index, umax = 7 : index, umin = 6 : index} : index
//       CHECK:  %[[A_CASTED:.*]] = arith.index_castui %[[A]] : index to i32
//       CHECK:  %[[B_CASTED:.*]] = arith.index_castui %[[B]] : index to i32
//       CHECK:  %[[RES:.*]] = arith.addi %[[A_CASTED]], %[[B_CASTED]] : i32
//       CHECK:  %[[RES_CASTED:.*]] = arith.index_castui %[[RES]] : i32 to index
//       CHECK:  return %[[RES_CASTED]] : index
func.func @test_addi() -> index {
  %0 = test.with_bounds { umin = 4 : index, umax = 5 : index, smin = 4 : index, smax = 5 : index } : index
  %1 = test.with_bounds { umin = 6 : index, umax = 7 : index, smin = 6 : index, smax = 7 : index } : index
  %2 = arith.addi %0, %1 : index
  return %2 : index
}


// CHECK-LABEL: func @test_addi_i64
//       CHECK:  %[[A:.*]] = test.with_bounds {smax = 5 : i64, smin = 4 : i64, umax = 5 : i64, umin = 4 : i64} : i64
//       CHECK:  %[[B:.*]] = test.with_bounds {smax = 7 : i64, smin = 6 : i64, umax = 7 : i64, umin = 6 : i64} : i64
//       CHECK:  %[[A_CASTED:.*]] = arith.trunci %[[A]] : i64 to i32
//       CHECK:  %[[B_CASTED:.*]] = arith.trunci %[[B]] : i64 to i32
//       CHECK:  %[[RES:.*]] = arith.addi %[[A_CASTED]], %[[B_CASTED]] : i32
//       CHECK:  %[[RES_CASTED:.*]] = arith.extui %[[RES]] : i32 to i64
//       CHECK:  return %[[RES_CASTED]] : i64
func.func @test_addi_i64() -> i64 {
  %0 = test.with_bounds { umin = 4 : i64, umax = 5 : i64, smin = 4 : i64, smax = 5 : i64 } : i64
  %1 = test.with_bounds { umin = 6 : i64, umax = 7 : i64, smin = 6 : i64, smax = 7 : i64 } : i64
  %2 = arith.addi %0, %1 : i64
  return %2 : i64
}

// CHECK-LABEL: func @test_cmpi
//       CHECK:  %[[A:.*]] = test.with_bounds {smax = 10 : index, smin = 0 : index, umax = 10 : index, umin = 0 : index} : index
//       CHECK:  %[[B:.*]] = test.with_bounds {smax = 10 : index, smin = 0 : index, umax = 10 : index, umin = 0 : index} : index
//       CHECK:  %[[A_CASTED:.*]] = arith.index_castui %[[A]] : index to i32
//       CHECK:  %[[B_CASTED:.*]] = arith.index_castui %[[B]] : index to i32
//       CHECK:  %[[RES:.*]] = arith.cmpi slt, %[[A_CASTED]], %[[B_CASTED]] : i32
//       CHECK:  return %[[RES]] : i1
func.func @test_cmpi() -> i1 {
  %0 = test.with_bounds { umin = 0 : index, umax = 10 : index, smin = 0 : index, smax = 10 : index } : index
  %1 = test.with_bounds { umin = 0 : index, umax = 10 : index, smin = 0 : index, smax = 10 : index } : index
  %2 = arith.cmpi slt, %0, %1 : index
  return %2 : i1
}
