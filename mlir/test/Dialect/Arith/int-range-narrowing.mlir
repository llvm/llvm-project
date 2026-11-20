// RUN: mlir-opt --arith-int-range-narrowing="int-bitwidths-supported=1,8,16,24,32" %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Some basic tests
//===----------------------------------------------------------------------===//

// Truncate possibly-negative values in a signed way
// CHECK-LABEL: func @test_addi_neg
//       CHECK:  %[[POS:.*]] = test.with_bounds <umin = 0 : index, umax = 1 : index, smin = 0 : index, smax = 1 : index> : index
//       CHECK:  %[[NEG:.*]] = test.with_bounds <umin = 0 : index, umax = -1 : index, smin = -1 : index, smax = 0 : index> : index
//       CHECK:  %[[POS_I8:.*]] = arith.index_castui %[[POS]] : index to i8
//       CHECK:  %[[NEG_I8:.*]] = arith.index_cast %[[NEG]] : index to i8
//       CHECK:  %[[RES_I8:.*]] = arith.addi %[[POS_I8]], %[[NEG_I8]] : i8
//       CHECK:  %[[RES:.*]] = arith.index_cast %[[RES_I8]] : i8 to index
//       CHECK:  return %[[RES]] : index
func.func @test_addi_neg() -> index {
  %0 = test.with_bounds < umin = 0 : index, umax = 1 : index, smin = 0 : index, smax = 1 : index > : index
  %1 = test.with_bounds < umin = 0 : index, umax = -1 : index, smin = -1 : index, smax = 0 : index > : index
  %2 = arith.addi %0, %1 : index
  return %2 : index
}

// CHECK-LABEL: func @test_addi
//       CHECK:  %[[A:.*]] = test.with_bounds <umin = 4 : index, umax = 5 : index, smin = 4 : index, smax = 5 : index> : index
//       CHECK:  %[[B:.*]] = test.with_bounds <umin = 6 : index, umax = 7 : index, smin = 6 : index, smax = 7 : index> : index
//       CHECK:  %[[A_CASTED:.*]] = arith.index_castui %[[A]] : index to i8
//       CHECK:  %[[B_CASTED:.*]] = arith.index_castui %[[B]] : index to i8
//       CHECK:  %[[RES:.*]] = arith.addi %[[A_CASTED]], %[[B_CASTED]] : i8
//       CHECK:  %[[RES_CASTED:.*]] = arith.index_castui %[[RES]] : i8 to index
//       CHECK:  return %[[RES_CASTED]] : index
func.func @test_addi() -> index {
  %0 = test.with_bounds < umin = 4 : index, umax = 5 : index, smin = 4 : index, smax = 5 : index > : index
  %1 = test.with_bounds < umin = 6 : index, umax = 7 : index, smin = 6 : index, smax = 7 : index > : index
  %2 = arith.addi %0, %1 : index
  return %2 : index
}

// CHECK-LABEL: func @test_addi_vec
//       CHECK:  %[[A:.*]] = test.with_bounds <umin = 4 : index, umax = 5 : index, smin = 4 : index, smax = 5 : index> : vector<4xindex>
//       CHECK:  %[[B:.*]] = test.with_bounds <umin = 6 : index, umax = 7 : index, smin = 6 : index, smax = 7 : index> : vector<4xindex>
//       CHECK:  %[[A_CASTED:.*]] = arith.index_castui %[[A]] : vector<4xindex> to vector<4xi8>
//       CHECK:  %[[B_CASTED:.*]] = arith.index_castui %[[B]] : vector<4xindex> to vector<4xi8>
//       CHECK:  %[[RES:.*]] = arith.addi %[[A_CASTED]], %[[B_CASTED]] : vector<4xi8>
//       CHECK:  %[[RES_CASTED:.*]] = arith.index_castui %[[RES]] : vector<4xi8> to vector<4xindex>
//       CHECK:  return %[[RES_CASTED]] : vector<4xindex>
func.func @test_addi_vec() -> vector<4xindex> {
  %0 = test.with_bounds < umin = 4 : index, umax = 5 : index, smin = 4 : index, smax = 5 : index > : vector<4xindex>
  %1 = test.with_bounds < umin = 6 : index, umax = 7 : index, smin = 6 : index, smax = 7 : index > : vector<4xindex>
  %2 = arith.addi %0, %1 : vector<4xindex>
  return %2 : vector<4xindex>
}

// CHECK-LABEL: func @test_addi_i64
//       CHECK:  %[[A:.*]] = test.with_bounds <umin = 4 : i64, umax = 5 : i64, smin = 4 : i64, smax = 5 : i64> : i64
//       CHECK:  %[[B:.*]] = test.with_bounds <umin = 6 : i64, umax = 7 : i64, smin = 6 : i64, smax = 7 : i64> : i64
//       CHECK:  %[[A_CASTED:.*]] = arith.trunci %[[A]] : i64 to i8
//       CHECK:  %[[B_CASTED:.*]] = arith.trunci %[[B]] : i64 to i8
//       CHECK:  %[[RES:.*]] = arith.addi %[[A_CASTED]], %[[B_CASTED]] : i8
//       CHECK:  %[[RES_CASTED:.*]] = arith.extui %[[RES]] : i8 to i64
//       CHECK:  return %[[RES_CASTED]] : i64
func.func @test_addi_i64() -> i64 {
  %0 = test.with_bounds < umin = 4 : i64, umax = 5 : i64, smin = 4 : i64, smax = 5 : i64 > : i64
  %1 = test.with_bounds < umin = 6 : i64, umax = 7 : i64, smin = 6 : i64, smax = 7 : i64 > : i64
  %2 = arith.addi %0, %1 : i64
  return %2 : i64
}

// CHECK-LABEL: func @test_cmpi
//       CHECK:  %[[A:.*]] = test.with_bounds <umin = 0 : index, umax = 10 : index, smin = 0 : index, smax = 10 : index> : index
//       CHECK:  %[[B:.*]] = test.with_bounds <umin = 0 : index, umax = 10 : index, smin = 0 : index, smax = 10 : index> : index
//       CHECK:  %[[A_CASTED:.*]] = arith.index_castui %[[A]] : index to i8
//       CHECK:  %[[B_CASTED:.*]] = arith.index_castui %[[B]] : index to i8
//       CHECK:  %[[RES:.*]] = arith.cmpi slt, %[[A_CASTED]], %[[B_CASTED]] : i8
//       CHECK:  return %[[RES]] : i1
func.func @test_cmpi() -> i1 {
  %0 = test.with_bounds < umin = 0 : index, umax = 10 : index, smin = 0 : index, smax = 10 : index > : index
  %1 = test.with_bounds < umin = 0 : index, umax = 10 : index, smin = 0 : index, smax = 10 : index > : index
  %2 = arith.cmpi slt, %0, %1 : index
  return %2 : i1
}

// CHECK-LABEL: func @test_cmpi_si_pred_out_of_signed_bounds
//       CHECK-NOT: arith.cmpi slt, {{.*}} : i32
//       CHECK-NOT: arith.cmpi sgt, {{.*}} : i32
//       CHECK-NOT: arith.cmpi sle, {{.*}} : i32
//       CHECK-NOT: arith.cmpi sge, {{.*}} : i32
//       CHECK:  %[[A:.*]] = test.with_bounds <umin = 0 : index, umax = 4292870144 : index, smin = 0 : index, smax = 4292870144 : index> : index
//       CHECK:  %[[B:.*]] = test.with_bounds <umin = 0 : index, umax = 4292870144 : index, smin = 0 : index, smax = 4292870144 : index> : index
//       CHECK:  %[[SLT:.*]] = arith.cmpi slt, %[[B]], %[[A]] : index
//       CHECK:  %[[C:.*]] = test.with_bounds <umin = 0 : index, umax = 2147483648 : index, smin = 0 : index, smax = 2147483648 : index> : index
//       CHECK:  %[[ZERO:.*]] = test.with_bounds <umin = 0 : index, umax = 0 : index, smin = 0 : index, smax = 0 : index> : index
//       CHECK:  %[[SGT:.*]] = arith.cmpi sgt, %[[C]], %[[ZERO]] : index
//       CHECK:  %[[SLE:.*]] = arith.cmpi sle, %[[A]], %[[ZERO]] : index
//       CHECK:  %[[SGE:.*]] = arith.cmpi sge, %[[C]], %[[ZERO]] : index
//       CHECK:  %[[AND0:.*]] = arith.andi %[[SLT]], %[[SGT]] : i1
//       CHECK:  %[[AND1:.*]] = arith.andi %[[SLE]], %[[SGE]] : i1
//       CHECK:  %[[AND2:.*]] = arith.andi %[[AND0]], %[[AND1]] : i1
//       CHECK:  return %[[AND2]] : i1
func.func @test_cmpi_si_pred_out_of_signed_bounds() -> i1 {
  %0 = test.with_bounds < umin = 0 : index, umax = 4292870144 : index, smin = 0 : index, smax = 4292870144 : index > : index
  %1 = test.with_bounds < umin = 0 : index, umax = 4292870144 : index, smin = 0 : index, smax = 4292870144 : index > : index
  %2 = arith.cmpi slt, %1, %0 : index
  %3 = test.with_bounds < umin = 0 : index, umax = 2147483648 : index, smin = 0 : index, smax = 2147483648 : index > : index
  %4 = test.with_bounds < umin = 0 : index, umax = 0 : index, smin = 0 : index, smax = 0 : index > : index
  %5 = arith.cmpi sgt, %3, %4 : index
  %6 = arith.cmpi sle, %0, %4 : index
  %7 = arith.cmpi sge, %3, %4 : index
  %8 = arith.andi %2, %5 : i1
  %9 = arith.andi %6, %7 : i1
  %10 = arith.andi %8, %9 : i1
  return %10 : i1
}

// CHECK-LABEL: func @test_cmpi_ui_pred_out_of_signed_bounds
//       CHECK:  %[[A:.*]] = test.with_bounds <umin = 0 : index, umax = 4292870144 : index, smin = 0 : index, smax = 4292870144 : index> : index
//       CHECK:  %[[B:.*]] = test.with_bounds <umin = 0 : index, umax = 4292870144 : index, smin = 0 : index, smax = 4292870144 : index> : index
//       CHECK:  %[[A_I32:.*]] = arith.index_castui %[[A]] : index to i32
//       CHECK:  %[[B_I32:.*]] = arith.index_castui %[[B]] : index to i32
//       CHECK:  %[[RES:.*]] = arith.cmpi ult, %[[A_I32]], %[[B_I32]] : i32
//       CHECK:  return %[[RES]] : i1
func.func @test_cmpi_ui_pred_out_of_signed_bounds() -> i1 {
  %0 = test.with_bounds < umin = 0 : index, umax = 4292870144 : index, smin = 0 : index, smax = 4292870144 : index > : index
  %1 = test.with_bounds < umin = 0 : index, umax = 4292870144 : index, smin = 0 : index, smax = 4292870144 : index > : index
  %2 = arith.cmpi ult, %0, %1 : index
  return %2 : i1
}

// CHECK-LABEL: func @test_cmpi_vec
//       CHECK:  %[[A:.*]] = test.with_bounds <umin = 0 : index, umax = 10 : index, smin = 0 : index, smax = 10 : index> : vector<4xindex>
//       CHECK:  %[[B:.*]] = test.with_bounds <umin = 0 : index, umax = 10 : index, smin = 0 : index, smax = 10 : index> : vector<4xindex>
//       CHECK:  %[[A_CASTED:.*]] = arith.index_castui %[[A]] : vector<4xindex> to vector<4xi8>
//       CHECK:  %[[B_CASTED:.*]] = arith.index_castui %[[B]] : vector<4xindex> to vector<4xi8>
//       CHECK:  %[[RES:.*]] = arith.cmpi slt, %[[A_CASTED]], %[[B_CASTED]] : vector<4xi8>
//       CHECK:  return %[[RES]] : vector<4xi1>
func.func @test_cmpi_vec() -> vector<4xi1> {
  %0 = test.with_bounds < umin = 0 : index, umax = 10 : index, smin = 0 : index, smax = 10 : index > : vector<4xindex>
  %1 = test.with_bounds < umin = 0 : index, umax = 10 : index, smin = 0 : index, smax = 10 : index > : vector<4xindex>
  %2 = arith.cmpi slt, %0, %1 : vector<4xindex>
  return %2 : vector<4xi1>
}

// CHECK-LABEL: func @test_add_cmpi
//       CHECK:  %[[A:.*]] = test.with_bounds <umin = 0 : index, umax = 10 : index, smin = 0 : index, smax = 10 : index> : index
//       CHECK:  %[[B:.*]] = test.with_bounds <umin = 0 : index, umax = 10 : index, smin = 0 : index, smax = 10 : index> : index
//       CHECK:  %[[C:.*]] = test.with_bounds <umin = 0 : index, umax = 10 : index, smin = 0 : index, smax = 10 : index> : index
//       CHECK:  %[[A_CASTED:.*]] = arith.index_castui %[[A]] : index to i8
//       CHECK:  %[[B_CASTED:.*]] = arith.index_castui %[[B]] : index to i8
//       CHECK:  %[[RES1:.*]] = arith.addi %[[A_CASTED]], %[[B_CASTED]] : i8
//       CHECK:  %[[C_CASTED:.*]] = arith.index_castui %[[C]] : index to i8
//       CHECK:  %[[RES2:.*]] = arith.cmpi slt, %[[C_CASTED]], %[[RES1]] : i8
//       CHECK:  return %[[RES2]] : i1
func.func @test_add_cmpi() -> i1 {
  %0 = test.with_bounds < umin = 0 : index, umax = 10 : index, smin = 0 : index, smax = 10 : index > : index
  %1 = test.with_bounds < umin = 0 : index, umax = 10 : index, smin = 0 : index, smax = 10 : index > : index
  %3 = test.with_bounds < umin = 0 : index, umax = 10 : index, smin = 0 : index, smax = 10 : index > : index
  %4 = arith.addi %0, %1 : index
  %5 = arith.cmpi slt, %3, %4 : index
  return %5 : i1
}

// CHECK-LABEL: func @test_add_cmpi_i64
//       CHECK:  %[[A:.*]] = test.with_bounds <umin = 0 : i64, umax = 10 : i64, smin = 0 : i64, smax = 10 : i64> : i64
//       CHECK:  %[[B:.*]] = test.with_bounds <umin = 0 : i64, umax = 10 : i64, smin = 0 : i64, smax = 10 : i64> : i64
//       CHECK:  %[[C:.*]] = test.with_bounds <umin = 0 : i64, umax = 10 : i64, smin = 0 : i64, smax = 10 : i64> : i64
//       CHECK:  %[[A_CASTED:.*]] = arith.trunci %[[A]] : i64 to i8
//       CHECK:  %[[B_CASTED:.*]] = arith.trunci %[[B]] : i64 to i8
//       CHECK:  %[[RES1:.*]] = arith.addi %[[A_CASTED]], %[[B_CASTED]] : i8
//       CHECK:  %[[C_CASTED:.*]] = arith.trunci %[[C]] : i64 to i8
//       CHECK:  %[[RES2:.*]] = arith.cmpi slt, %[[C_CASTED]], %[[RES1]] : i8
//       CHECK:  return %[[RES2]] : i1
func.func @test_add_cmpi_i64() -> i1 {
  %0 = test.with_bounds < umin = 0 : i64, umax = 10 : i64, smin = 0 : i64, smax = 10 : i64 > : i64
  %1 = test.with_bounds < umin = 0 : i64, umax = 10 : i64, smin = 0 : i64, smax = 10 : i64 > : i64
  %3 = test.with_bounds < umin = 0 : i64, umax = 10 : i64, smin = 0 : i64, smax = 10 : i64 > : i64
  %4 = arith.addi %0, %1 : i64
  %5 = arith.cmpi slt, %3, %4 : i64
  return %5 : i1
}

//===----------------------------------------------------------------------===//
// arith.addi
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @addi_extui_i8
// CHECK-SAME:    (%[[ARG0:.+]]: i8, %[[ARG1:.+]]: i8)
// CHECK-NEXT:    %[[EXT0:.+]] = arith.extui %[[ARG0]] : i8 to i32
// CHECK-NEXT:    %[[EXT1:.+]] = arith.extui %[[ARG1]] : i8 to i32
// CHECK-NEXT:    %[[LHS:.+]]  = arith.trunci %[[EXT0]] : i32 to i16
// CHECK-NEXT:    %[[RHS:.+]]  = arith.trunci %[[EXT1]] : i32 to i16
// CHECK-NEXT:    %[[ADD:.+]]  = arith.addi %[[LHS]], %[[RHS]] : i16
// CHECK-NEXT:    %[[RET:.+]]  = arith.extui %[[ADD]] : i16 to i32
// CHECK-NEXT:    return %[[RET]] : i32
func.func @addi_extui_i8(%lhs: i8, %rhs: i8) -> i32 {
  %a = arith.extui %lhs : i8 to i32
  %b = arith.extui %rhs : i8 to i32
  %r = arith.addi %a, %b : i32
  return %r : i32
}

// This can be optimized to i16 since we're dealing in [-128, 127] + [0, 255],
// which is [-128, 382]
//
// CHECK-LABEL: func.func @addi_mixed_ext_i8
// CHECK-SAME:    (%[[ARG0:.+]]: i8, %[[ARG1:.+]]: i8)
// CHECK-NEXT:    %[[EXT0:.+]] = arith.extsi %[[ARG0]] : i8 to i32
// CHECK-NEXT:    %[[EXT1:.+]] = arith.extui %[[ARG1]] : i8 to i32
// CHECK-NEXT:    %[[LHS:.+]]  = arith.trunci %[[EXT0]] : i32 to i16
// CHECK-NEXT:    %[[RHS:.+]]  = arith.trunci %[[EXT1]] : i32 to i16
// CHECK-NEXT:    %[[ADD:.+]]  = arith.addi %[[LHS]], %[[RHS]] : i16
// CHECK-NEXT:    %[[RET:.+]]  = arith.extsi %[[ADD]] : i16 to i32
// CHECK-NEXT:    return %[[RET]] : i32
func.func @addi_mixed_ext_i8(%lhs: i8, %rhs: i8) -> i32 {
  %a = arith.extsi %lhs : i8 to i32
  %b = arith.extui %rhs : i8 to i32
  %r = arith.addi %a, %b : i32
  return %r : i32
}

// This case should not get optimized because we cannot reduce the bitwidth
// below i16, given the pass options set.
//
// CHECK-LABEL: func.func @addi_extsi_i16
// CHECK-SAME:    (%[[ARG0:.+]]: i8, %[[ARG1:.+]]: i8)
// CHECK-NEXT:    %[[EXT0:.+]] = arith.extsi %[[ARG0]] : i8 to i16
// CHECK-NEXT:    %[[EXT1:.+]] = arith.extsi %[[ARG1]] : i8 to i16
// CHECK-NEXT:    %[[ADD:.+]]  = arith.addi %[[EXT0]], %[[EXT1]] : i16
// CHECK-NEXT:    return %[[ADD]] : i16
func.func @addi_extsi_i16(%lhs: i8, %rhs: i8) -> i16 {
  %a = arith.extsi %lhs : i8 to i16
  %b = arith.extsi %rhs : i8 to i16
  %r = arith.addi %a, %b : i16
  return %r : i16
}

//===----------------------------------------------------------------------===//
// arith.subi
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @subi_extui_i8
// CHECK-SAME:    (%[[ARG0:.+]]: i8, %[[ARG1:.+]]: i8)
// CHECK-NEXT:    %[[EXT0:.+]] = arith.extui %[[ARG0]] : i8 to i32
// CHECK-NEXT:    %[[EXT1:.+]] = arith.extui %[[ARG1]] : i8 to i32
// CHECK-NEXT:    %[[LHS:.+]]  = arith.trunci %[[EXT0]] : i32 to i16
// CHECK-NEXT:    %[[RHS:.+]]  = arith.trunci %[[EXT1]] : i32 to i16
// CHECK-NEXT:    %[[SUB:.+]]  = arith.subi %[[LHS]], %[[RHS]] : i16
// CHECK-NEXT:    %[[RET:.+]]  = arith.extsi %[[SUB]] : i16 to i32
// CHECK-NEXT:    return %[[RET]] : i32
func.func @subi_extui_i8(%lhs: i8, %rhs: i8) -> i32 {
  %a = arith.extui %lhs : i8 to i32
  %b = arith.extui %rhs : i8 to i32
  %r = arith.subi %a, %b : i32
  return %r : i32
}

// Despite the mixed sign and zero extensions, we can optimize here
//
// CHECK-LABEL: func.func @subi_mixed_ext_i8
// CHECK-SAME:    (%[[ARG0:.+]]: i8, %[[ARG1:.+]]: i8)
// CHECK-NEXT:    %[[EXT0:.+]] = arith.extsi %[[ARG0]] : i8 to i32
// CHECK-NEXT:    %[[EXT1:.+]] = arith.extui %[[ARG1]] : i8 to i32
// CHECK-NEXT:    %[[LHS:.+]]  = arith.trunci %[[EXT0]] : i32 to i16
// CHECK-NEXT:    %[[RHS:.+]]  = arith.trunci %[[EXT1]] : i32 to i16
// CHECK-NEXT:    %[[ADD:.+]]  = arith.subi %[[LHS]], %[[RHS]] : i16
// CHECK-NEXT:    %[[RET:.+]]  = arith.extsi %[[ADD]] : i16 to i32
// CHECK-NEXT:    return %[[RET]] : i32
func.func @subi_mixed_ext_i8(%lhs: i8, %rhs: i8) -> i32 {
  %a = arith.extsi %lhs : i8 to i32
  %b = arith.extui %rhs : i8 to i32
  %r = arith.subi %a, %b : i32
  return %r : i32
}

// CHECK-LABEL: func.func @signed_ops_out_of_narrowed_signed_range
// CHECK-NOT: arith.divsi {{.*}} : i32
// CHECK-NOT: arith.ceildivsi {{.*}} : i32
// CHECK-NOT: arith.floordivsi {{.*}} : i32
// CHECK-NOT: arith.remsi {{.*}} : i32
// CHECK-NOT: arith.maxsi {{.*}} : i32
// CHECK-NOT: arith.minsi {{.*}} : i32
// CHECK-NOT: arith.shrsi {{.*}} : i32
// CHECK: %[[DIV_I64:.*]] = arith.divsi {{.*}} : i64
// CHECK: %[[CEIL_I64:.*]] = arith.ceildivsi {{.*}} : i64
// CHECK: %[[FLOOR_I64:.*]] = arith.floordivsi {{.*}} : i64
// CHECK: %[[REM_I64:.*]] = arith.remsi {{.*}} : i64
// CHECK: %[[MAX_I64:.*]] = arith.maxsi {{.*}} : i64
// CHECK: %[[MIN_I64:.*]] = arith.minsi {{.*}} : i64
// CHECK: %[[SHR_I64:.*]] = arith.shrsi {{.*}} : i64
// CHECK: return %{{.*}} : i64, i64, i64, i64, i64, i64, i64
func.func @signed_ops_out_of_narrowed_signed_range() -> (i64, i64, i64, i64, i64, i64, i64) {
  %0 = test.with_bounds < umin = 0 : i64, umax = 4292870144 : i64, smin = 0 : i64, smax = 4292870144 : i64 > : i64
  %1 = test.with_bounds < umin = 1 : i64, umax = 8 : i64, smin = 1 : i64, smax = 8 : i64 > : i64
  %2 = test.with_bounds < umin = 0 : i64, umax = 0 : i64, smin = 0 : i64, smax = 0 : i64 > : i64
  %3 = arith.divsi %0, %1 : i64
  %4 = arith.ceildivsi %0, %1 : i64
  %5 = arith.floordivsi %0, %1 : i64
  %6 = arith.remsi %0, %1 : i64
  %7 = arith.maxsi %0, %2 : i64
  %8 = arith.minsi %0, %2 : i64
  %9 = arith.shrsi %0, %1 : i64
  return %3, %4, %5, %6, %7, %8, %9 : i64, i64, i64, i64, i64, i64, i64
}

// CHECK-LABEL: func.func @unsigned_ops_out_of_narrowed_signed_range
// CHECK: arith.divui {{.*}} : i32
// CHECK: arith.ceildivui {{.*}} : i32
// CHECK: arith.remui {{.*}} : i32
// CHECK: arith.maxui {{.*}} : i32
// CHECK: arith.minui {{.*}} : i32
// CHECK: arith.shrui {{.*}} : i32
// CHECK: return %{{.*}} : i64, i64, i64, i64, i64, i64
func.func @unsigned_ops_out_of_narrowed_signed_range() -> (i64, i64, i64, i64, i64, i64) {
  %0 = test.with_bounds < umin = 0 : i64, umax = 4292870144 : i64, smin = 0 : i64, smax = 4292870144 : i64 > : i64
  %1 = test.with_bounds < umin = 1 : i64, umax = 8 : i64, smin = 1 : i64, smax = 8 : i64 > : i64
  %2 = test.with_bounds < umin = 0 : i64, umax = 0 : i64, smin = 0 : i64, smax = 0 : i64 > : i64
  %3 = arith.divui %0, %1 : i64
  %4 = arith.ceildivui %0, %1 : i64
  %5 = arith.remui %0, %1 : i64
  %6 = arith.maxui %0, %2 : i64
  %7 = arith.minui %0, %2 : i64
  %8 = arith.shrui %0, %1 : i64
  return %3, %4, %5, %6, %7, %8 : i64, i64, i64, i64, i64, i64
}

//===----------------------------------------------------------------------===//
// Unsigned division, remainder and logical right shift
//===----------------------------------------------------------------------===//

// For x = -3, the i9 operation computes 509 / 7 = 72, but narrowing gives
// 253 / 7 = 36. Both results fit signed i8, so result ranges do not prevent it.
// CHECK-LABEL: func.func @divui_extsi_i8
// CHECK-SAME: (%[[ARG:.+]]: i8)
// CHECK-DAG: %[[C7:.+]] = arith.constant 7 : i9
// CHECK-DAG: %[[WIDE:.+]] = arith.extsi %[[ARG]] : i8 to i9
// CHECK: %[[DIV:.+]] = arith.divui %[[WIDE]], %[[C7]] : i9
// CHECK-NEXT: return %[[DIV]] : i9
func.func @divui_extsi_i8(%x: i8) -> i9 {
  %c7 = arith.constant 7 : i9
  %wide = arith.extsi %x : i8 to i9
  %div = arith.divui %wide, %c7 : i9
  return %div : i9
}

// For x = -3, the i9 operation computes ceil(509 / 7) = 73, but narrowing
// gives ceil(253 / 7) = 37. Both results fit signed i8.
// CHECK-LABEL: func.func @ceildivui_extsi_i8
// CHECK-SAME: (%[[ARG:.+]]: i8)
// CHECK-DAG: %[[C7:.+]] = arith.constant 7 : i9
// CHECK-DAG: %[[WIDE:.+]] = arith.extsi %[[ARG]] : i8 to i9
// CHECK: %[[DIV:.+]] = arith.ceildivui %[[WIDE]], %[[C7]] : i9
// CHECK-NEXT: return %[[DIV]] : i9
func.func @ceildivui_extsi_i8(%x: i8) -> i9 {
  %c7 = arith.constant 7 : i9
  %wide = arith.extsi %x : i8 to i9
  %div = arith.ceildivui %wide, %c7 : i9
  return %div : i9
}

// Signed representability is insufficient for unsigned remainder. For x = -3,
// the i16 operation computes 65533 % 7 = 6, but narrowing gives 253 % 7 = 1.
// CHECK-LABEL: func.func @remui_extsi_i8
// CHECK-SAME: (%[[ARG:.+]]: i8)
// CHECK-DAG: %[[C7:.+]] = arith.constant 7 : i16
// CHECK-DAG: %[[WIDE:.+]] = arith.extsi %[[ARG]] : i8 to i16
// CHECK: %[[REM:.+]] = arith.remui %[[WIDE]], %[[C7]] : i16
// CHECK-NEXT: return %[[REM]] : i16
func.func @remui_extsi_i8(%x: i8) -> i16 {
  %c7 = arith.constant 7 : i16
  %wide = arith.extsi %x : i8 to i16
  %rem = arith.remui %wide, %c7 : i16
  return %rem : i16
}

// Zero extension preserves both unsigned operands, including values >= 128.
// CHECK-LABEL: func.func @remui_extui_i8
// CHECK-SAME: (%[[LHS:.+]]: i8, %[[RHS:.+]]: i8)
// CHECK-NEXT: %[[REM:.+]] = arith.remui %[[LHS]], %[[RHS]] : i8
// CHECK-NEXT: %[[RESULT:.+]] = arith.extui %[[REM]] : i8 to i16
// CHECK-NEXT: return %[[RESULT]] : i16
func.func @remui_extui_i8(%lhs: i8, %rhs: i8) -> i16 {
  %a = arith.extui %lhs : i8 to i16
  %b = arith.extui %rhs : i8 to i16
  %rem = arith.remui %a, %b : i16
  return %rem : i16
}

// A sign-extended value can still narrow when its range proves it nonnegative.
// CHECK-LABEL: func.func @remui_nonnegative_extsi_i8
// CHECK-SAME: (%[[ARG:.+]]: i8)
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : i8
// CHECK-DAG: %[[C7:.+]] = arith.constant 7 : i8
// CHECK: %[[NONNEGATIVE:.+]] = arith.maxsi %[[ARG]], %[[C0]] : i8
// CHECK-NEXT: %[[REM:.+]] = arith.remui %[[NONNEGATIVE]], %[[C7]] : i8
// CHECK-NEXT: %[[RESULT:.+]] = arith.extui %[[REM]] : i8 to i16
// CHECK-NEXT: return %[[RESULT]] : i16
func.func @remui_nonnegative_extsi_i8(%x: i8) -> i16 {
  %c0 = arith.constant 0 : i8
  %c7 = arith.constant 7 : i16
  %nonnegative = arith.maxsi %x, %c0 : i8
  %wide = arith.extsi %nonnegative : i8 to i16
  %rem = arith.remui %wide, %c7 : i16
  return %rem : i16
}

// For x = -3, the i9 operation computes 509 >> 2 = 127, but narrowing gives
// 253 >> 2 = 63. The shift amount is also below the narrowed bitwidth.
// CHECK-LABEL: func.func @shrui_extsi_i8
// CHECK-SAME: (%[[ARG:.+]]: i8)
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : i9
// CHECK-DAG: %[[WIDE:.+]] = arith.extsi %[[ARG]] : i8 to i9
// CHECK: %[[SHIFT:.+]] = arith.shrui %[[WIDE]], %[[C2]] : i9
// CHECK-NEXT: return %[[SHIFT]] : i9
func.func @shrui_extsi_i8(%x: i8) -> i9 {
  %c2 = arith.constant 2 : i9
  %wide = arith.extsi %x : i8 to i9
  %shift = arith.shrui %wide, %c2 : i9
  return %shift : i9
}

//===----------------------------------------------------------------------===//
// Shift ops must not be narrowed when the shift amount can reach the target
// bitwidth, since a shift >= the bitwidth is poison. Here the value fits in
// i32 but the amount [0, 63] does not, so narrowing must be skipped.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @shrsi_amount_out_of_narrowed_range
// CHECK: arith.shrsi {{.*}} : i64
// CHECK-NOT: arith.shrsi {{.*}} : i32
func.func @shrsi_amount_out_of_narrowed_range() -> i64 {
  %0 = test.with_bounds < umin = 0 : i64, umax = 4 : i64, smin = 0 : i64, smax = 4 : i64 > : i64
  %1 = test.with_bounds < umin = 0 : i64, umax = 63 : i64, smin = 0 : i64, smax = 63 : i64 > : i64
  %2 = arith.shrsi %0, %1 : i64
  return %2 : i64
}

// When the amount is provably below the target bitwidth, narrowing is safe.
//
// CHECK-LABEL: func.func @shrui_amount_in_narrowed_range
// CHECK: arith.shrui {{.*}} : i32
func.func @shrui_amount_in_narrowed_range() -> i64 {
  %0 = test.with_bounds < umin = 0 : i64, umax = 4 : i64, smin = 0 : i64, smax = 4 : i64 > : i64
  %1 = test.with_bounds < umin = 0 : i64, umax = 31 : i64, smin = 0 : i64, smax = 31 : i64 > : i64
  %2 = arith.shrui %0, %1 : i64
  return %2 : i64
}

//===----------------------------------------------------------------------===//
// arith.muli
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @muli_extui_i8
// CHECK-SAME:    (%[[ARG0:.+]]: i8, %[[ARG1:.+]]: i8)
// CHECK-NEXT:    %[[EXT0:.+]] = arith.extui %[[ARG0]] : i8 to i32
// CHECK-NEXT:    %[[EXT1:.+]] = arith.extui %[[ARG1]] : i8 to i32
// CHECK-NEXT:    %[[LHS:.+]]  = arith.trunci %[[EXT0]] : i32 to i16
// CHECK-NEXT:    %[[RHS:.+]]  = arith.trunci %[[EXT1]] : i32 to i16
// CHECK-NEXT:    %[[MUL:.+]]  = arith.muli %[[LHS]], %[[RHS]] : i16
// CHECK-NEXT:    %[[RET:.+]]  = arith.extui %[[MUL]] : i16 to i32
// CHECK-NEXT:    return %[[RET]] : i32
func.func @muli_extui_i8(%lhs: i8, %rhs: i8) -> i32 {
  %a = arith.extui %lhs : i8 to i32
  %b = arith.extui %rhs : i8 to i32
  %r = arith.muli %a, %b : i32
  return %r : i32
}

// We do not expect this case to be optimized because given n-bit operands,
// arith.muli produces 2n bits of result.
//
// CHECK-LABEL: func.func @muli_extsi_i32
// CHECK-SAME:    (%[[ARG0:.+]]: i16, %[[ARG1:.+]]: i16)
// CHECK-NEXT:    %[[LHS:.+]]  = arith.extsi %[[ARG0]] : i16 to i32
// CHECK-NEXT:    %[[RHS:.+]]  = arith.extsi %[[ARG1]] : i16 to i32
// CHECK-NEXT:    %[[RET:.+]]  = arith.muli %[[LHS]], %[[RHS]] : i32
// CHECK-NEXT:    return %[[RET]] : i32
func.func @muli_extsi_i32(%lhs: i16, %rhs: i16) -> i32 {
  %a = arith.extsi %lhs : i16 to i32
  %b = arith.extsi %rhs : i16 to i32
  %r = arith.muli %a, %b : i32
  return %r : i32
}

// The mixed extensions mean that we have [-128, 127] * [0, 255], which can
// be computed exactly in i16.
//
// CHECK-LABEL: func.func @muli_mixed_ext_i8
// CHECK-SAME:    (%[[ARG0:.+]]: i8, %[[ARG1:.+]]: i8)
// CHECK-NEXT:    %[[EXT0:.+]] = arith.extsi %[[ARG0]] : i8 to i32
// CHECK-NEXT:    %[[EXT1:.+]] = arith.extui %[[ARG1]] : i8 to i32
// CHECK-NEXT:    %[[LHS:.+]]  = arith.trunci %[[EXT0]] : i32 to i16
// CHECK-NEXT:    %[[RHS:.+]]  = arith.trunci %[[EXT1]] : i32 to i16
// CHECK-NEXT:    %[[MUL:.+]]  = arith.muli %[[LHS]], %[[RHS]] : i16
// CHECK-NEXT:    %[[RET:.+]]  = arith.extsi %[[MUL]] : i16 to i32
// CHECK-NEXT:    return %[[RET]] : i32
func.func @muli_mixed_ext_i8(%lhs: i8, %rhs: i8) -> i32 {
  %a = arith.extsi %lhs : i8 to i32
  %b = arith.extui %rhs : i8 to i32
  %r = arith.muli %a, %b : i32
  return %r : i32
}

//===----------------------------------------------------------------------===//
// Overflow flags must be valid at the narrowed width.
//===----------------------------------------------------------------------===//

// For x = 127, i16 127 + 1 = 128 is defined. Keeping nsw on the i8 addition
// would produce poison, even though the result fits unsigned i8.
// CHECK-LABEL: func.func @addi_narrow_drop_nsw
// CHECK-SAME: (%[[ARG:.+]]: i8)
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : i8
// CHECK-DAG: %[[C130:.+]] = arith.constant -126 : i8
// CHECK: %[[BOUNDED:.+]] = arith.minui %[[ARG]], %[[C130]] : i8
// CHECK-NEXT: %[[SUM:.+]] = arith.addi %[[BOUNDED]], %[[C1]] : i8
// CHECK-NEXT: %[[RESULT:.+]] = arith.extui %[[SUM]] : i8 to i16
// CHECK-NEXT: return %[[RESULT]] : i16
func.func @addi_narrow_drop_nsw(%x: i8) -> i16 {
  %wide = arith.extui %x : i8 to i16
  %c130 = arith.constant 130 : i16
  %bounded = arith.minui %wide, %c130 : i16
  %c1 = arith.constant 1 : i16
  %sum = arith.addi %bounded, %c1 overflow<nsw> : i16
  return %sum : i16
}

// For x = 128, i16 128 - 1 = 127 is defined, but signed i8 -128 - 1
// overflows. Drop nsw while retaining the valid nuw flag.
// CHECK-LABEL: func.func @subi_narrow_drop_nsw
// CHECK-SAME: (%[[ARG:.+]]: i8)
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : i8
// CHECK-DAG: %[[C128:.+]] = arith.constant -128 : i8
// CHECK: %[[BOUNDED:.+]] = arith.maxui %[[ARG]], %[[C128]] : i8
// CHECK-NEXT: %[[DIFF:.+]] = arith.subi %[[BOUNDED]], %[[C1]] overflow<nuw> : i8
// CHECK-NEXT: %[[RESULT:.+]] = arith.extui %[[DIFF]] : i8 to i16
// CHECK-NEXT: return %[[RESULT]] : i16
func.func @subi_narrow_drop_nsw(%x: i8) -> i16 {
  %wide = arith.extui %x : i8 to i16
  %c128 = arith.constant 128 : i16
  %bounded = arith.maxui %wide, %c128 : i16
  %c1 = arith.constant 1 : i16
  %diff = arith.subi %bounded, %c1 overflow<nsw, nuw> : i16
  return %diff : i16
}

// For x = 64, i16 64 * 2 = 128 is defined. Keeping nsw on the i8
// multiplication would produce poison.
// CHECK-LABEL: func.func @muli_narrow_drop_nsw
// CHECK-SAME: (%[[ARG:.+]]: i8)
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : i8
// CHECK-DAG: %[[C127:.+]] = arith.constant 127 : i8
// CHECK: %[[BOUNDED:.+]] = arith.minui %[[ARG]], %[[C127]] : i8
// CHECK-NEXT: %[[PRODUCT:.+]] = arith.muli %[[BOUNDED]], %[[C2]] : i8
// CHECK-NEXT: %[[RESULT:.+]] = arith.extui %[[PRODUCT]] : i8 to i16
// CHECK-NEXT: return %[[RESULT]] : i16
func.func @muli_narrow_drop_nsw(%x: i8) -> i16 {
  %wide = arith.extui %x : i8 to i16
  %c127 = arith.constant 127 : i16
  %bounded = arith.minui %wide, %c127 : i16
  %c2 = arith.constant 2 : i16
  %product = arith.muli %bounded, %c2 overflow<nsw> : i16
  return %product : i16
}

// For x = 64, i16 64 << 1 = 128 is defined, but the i8 shift overflows
// signed arithmetic. Drop nsw while retaining the valid nuw flag.
// CHECK-LABEL: func.func @shli_narrow_drop_nsw
// CHECK-SAME: (%[[ARG:.+]]: i8)
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : i8
// CHECK-DAG: %[[C127:.+]] = arith.constant 127 : i8
// CHECK: %[[BOUNDED:.+]] = arith.minui %[[ARG]], %[[C127]] : i8
// CHECK-NEXT: %[[SHIFT:.+]] = arith.shli %[[BOUNDED]], %[[C1]] overflow<nuw> : i8
// CHECK-NEXT: %[[RESULT:.+]] = arith.extui %[[SHIFT]] : i8 to i16
// CHECK-NEXT: return %[[RESULT]] : i16
func.func @shli_narrow_drop_nsw(%x: i8) -> i16 {
  %wide = arith.extui %x : i8 to i16
  %c127 = arith.constant 127 : i16
  %bounded = arith.minui %wide, %c127 : i16
  %c1 = arith.constant 1 : i16
  %shift = arith.shli %bounded, %c1 overflow<nsw, nuw> : i16
  return %shift : i16
}

// For x = 22, i9 22 * 23 = 506 (signed -6) is defined with nuw. The result
// fits signed i8, but keeping nuw on i8 22 * 23 would produce poison.
// CHECK-LABEL: func.func @muli_narrow_drop_nuw
// CHECK-SAME: (%[[ARG:.+]]: i8)
// CHECK-DAG: %[[C21:.+]] = arith.constant 21 : i8
// CHECK-DAG: %[[C22:.+]] = arith.constant 22 : i8
// CHECK-DAG: %[[C23:.+]] = arith.constant 23 : i8
// CHECK: %[[LOWER:.+]] = arith.maxui %[[ARG]], %[[C21]] : i8
// CHECK-NEXT: %[[BOUNDED:.+]] = arith.minui %[[LOWER]], %[[C22]] : i8
// CHECK-NEXT: %[[PRODUCT:.+]] = arith.muli %[[BOUNDED]], %[[C23]] : i8
// CHECK-NEXT: %[[RESULT:.+]] = arith.extsi %[[PRODUCT]] : i8 to i9
// CHECK-NEXT: return %[[RESULT]] : i9
func.func @muli_narrow_drop_nuw(%x: i8) -> i9 {
  %c21 = arith.constant 21 : i8
  %c22 = arith.constant 22 : i8
  %lower = arith.maxui %x, %c21 : i8
  %bounded = arith.minui %lower, %c22 : i8
  %wide = arith.extui %bounded : i8 to i9
  %c23 = arith.constant 23 : i9
  %product = arith.muli %wide, %c23 overflow<nuw> : i9
  return %product : i9
}

// For x = 63, i9 63 << 3 = 504 (signed -8) is defined with nuw. The result
// fits signed i8, but keeping nuw on i8 63 << 3 would produce poison.
// CHECK-LABEL: func.func @shli_narrow_drop_nuw
// CHECK-SAME: (%[[ARG:.+]]: i8)
// CHECK-DAG: %[[C3:.+]] = arith.constant 3 : i8
// CHECK-DAG: %[[C60:.+]] = arith.constant 60 : i8
// CHECK-DAG: %[[C63:.+]] = arith.constant 63 : i8
// CHECK: %[[LOWER:.+]] = arith.maxui %[[ARG]], %[[C60]] : i8
// CHECK-NEXT: %[[BOUNDED:.+]] = arith.minui %[[LOWER]], %[[C63]] : i8
// CHECK-NEXT: %[[SHIFT:.+]] = arith.shli %[[BOUNDED]], %[[C3]] : i8
// CHECK-NEXT: %[[RESULT:.+]] = arith.extsi %[[SHIFT]] : i8 to i9
// CHECK-NEXT: return %[[RESULT]] : i9
func.func @shli_narrow_drop_nuw(%x: i8) -> i9 {
  %c60 = arith.constant 60 : i8
  %c63 = arith.constant 63 : i8
  %lower = arith.maxui %x, %c60 : i8
  %bounded = arith.minui %lower, %c63 : i8
  %wide = arith.extui %bounded : i8 to i9
  %c3 = arith.constant 3 : i9
  %shift = arith.shli %wide, %c3 overflow<nuw> : i9
  return %shift : i9
}

// Signed narrowing preserves nsw when the operands and result fit signed i8.
// CHECK-LABEL: func.func @addi_narrow_preserve_nsw
// CHECK-SAME: (%[[ARG:.+]]: i8)
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : i8
// CHECK-DAG: %[[CM1:.+]] = arith.constant -1 : i8
// CHECK: %[[BOUNDED:.+]] = arith.minsi %[[ARG]], %[[CM1]] : i8
// CHECK-NEXT: %[[SUM:.+]] = arith.addi %[[BOUNDED]], %[[C1]] overflow<nsw> : i8
// CHECK-NEXT: %[[RESULT:.+]] = arith.extsi %[[SUM]] : i8 to i16
// CHECK-NEXT: return %[[RESULT]] : i16
func.func @addi_narrow_preserve_nsw(%x: i8) -> i16 {
  %wide = arith.extsi %x : i8 to i16
  %cm1 = arith.constant -1 : i16
  %bounded = arith.minsi %wide, %cm1 : i16
  %c1 = arith.constant 1 : i16
  %sum = arith.addi %bounded, %c1 overflow<nsw> : i16
  return %sum : i16
}

// Both flags remain valid when both signed and unsigned values are preserved.
// CHECK-LABEL: func.func @addi_narrow_preserve_both
// CHECK-SAME: (%[[ARG:.+]]: i8)
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : i8
// CHECK-DAG: %[[C63:.+]] = arith.constant 63 : i8
// CHECK: %[[BOUNDED:.+]] = arith.minui %[[ARG]], %[[C63]] : i8
// CHECK-NEXT: %[[SUM:.+]] = arith.addi %[[BOUNDED]], %[[C1]] overflow<nsw, nuw> : i8
// CHECK-NEXT: %[[RESULT:.+]] = arith.extui %[[SUM]] : i8 to i16
// CHECK-NEXT: return %[[RESULT]] : i16
func.func @addi_narrow_preserve_both(%x: i8) -> i16 {
  %wide = arith.extui %x : i8 to i16
  %c63 = arith.constant 63 : i16
  %bounded = arith.minui %wide, %c63 : i16
  %c1 = arith.constant 1 : i16
  %sum = arith.addi %bounded, %c1 overflow<nsw, nuw> : i16
  return %sum : i16
}

// Can't reduce width here since we need the extra bits
// CHECK-LABEL: func.func @i32_overflows_to_index
// CHECK-SAME: (%[[ARG0:.+]]: i32)
// CHECK: %[[CLAMPED:.+]] = arith.maxsi %[[ARG0]], %{{.*}} : i32
// CHECK: %[[CAST:.+]] = arith.index_castui %[[CLAMPED]] : i32 to index
// CHECK: %[[MUL:.+]] = arith.muli %[[CAST]], %{{.*}} : index
// CHECK: return %[[MUL]] : index
func.func @i32_overflows_to_index(%arg0: i32) -> index {
  %c0_i32 = arith.constant 0 : i32
  %c4 = arith.constant 4 : index
  %clamped = arith.maxsi %arg0, %c0_i32 : i32
  %cast = arith.index_castui %clamped : i32 to index
  %mul = arith.muli %cast, %c4 : index
  return %mul : index
}

// Can't reduce width here since we need the extra bits
// CHECK-LABEL: func.func @i32_overflows_to_i64
// CHECK-SAME: (%[[ARG0:.+]]: i32)
// CHECK: %[[CLAMPED:.+]] = arith.maxsi %[[ARG0]], %{{.*}} : i32
// CHECK: %[[CAST:.+]] = arith.extui %[[CLAMPED]] : i32 to i64
// CHECK: %[[MUL:.+]] = arith.muli %[[CAST]], %{{.*}} : i64
// CHECK: return %[[MUL]] : i64
func.func @i32_overflows_to_i64(%arg0: i32) -> i64 {
  %c0_i32 = arith.constant 0 : i32
  %c4_i64 = arith.constant 4 : i64
  %clamped = arith.maxsi %arg0, %c0_i32 : i32
  %cast = arith.extui %clamped : i32 to i64
  %mul = arith.muli %cast, %c4_i64 : i64
  return %mul : i64
}

// Motivating example for negative number support, added as a test case
// and simplified
// CHECK-LABEL: func.func @clamp_to_loop_bound_and_id()
// CHECK-DAG: %[[C64_I8:.+]] = arith.constant 64 : i8
// CHECK-DAG: %[[C64_I16:.+]] = arith.constant 64 : i16
// CHECK-DAG: %[[C16_I16:.+]] = arith.constant 16 : i16
// CHECK-DAG: %[[C0_I16:.+]] = arith.constant 0 : i16
// CHECK: %[[TID:.+]] = test.with_bounds
// CHECK-SAME: umax = 63
// CHECK: %[[BOUND:.+]] = test.with_bounds
// CHECK-SAME: umax = 112
// Loop narrows to i16 (not i8) because indVar+step=[80,144] doesn't fit in signed i8.
// CHECK: %[[BOUND_I16:.+]] = arith.index_castui %[[BOUND]] : index to i16
// CHECK: scf.for %[[ARG0:.+]] = %[[C16_I16]] to %[[BOUND_I16]] step %[[C64_I16]]  : i16 {
// CHECK:   %[[ARG0_INDEX:.+]] = arith.index_castui %[[ARG0]] : i16 to index
// CHECK:   %[[BOUND_I8:.+]] = arith.index_castui %[[BOUND]] : index to i8
// CHECK:   %[[ARG0_I8:.+]] = arith.index_castui %[[ARG0_INDEX]] : index to i8
// CHECK:   %[[V0_I8:.+]] = arith.subi %[[BOUND_I8]], %[[ARG0_I8]] : i8
// CHECK:   %[[V1_I8:.+]] = arith.minsi %[[V0_I8]], %[[C64_I8]] : i8
// CHECK:   %[[V1_INDEX:.+]] = arith.index_cast %[[V1_I8]] : i8 to index
// CHECK:   %[[V1_I16:.+]] = arith.index_cast %[[V1_INDEX]] : index to i16
// CHECK:   %[[TID_I16:.+]] = arith.index_castui %[[TID]] : index to i16
// CHECK:   %[[V2_I16:.+]] = arith.subi %[[V1_I16]], %[[TID_I16]] : i16
// CHECK:   %[[V3:.+]] = arith.cmpi slt, %[[V2_I16]], %[[C0_I16]] : i16
// CHECK:   scf.if %[[V3]]
func.func @clamp_to_loop_bound_and_id() {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c64 = arith.constant 64 : index

  %tid = test.with_bounds <smin = 0 : index, smax = 63 : index, umin = 0 : index, umax = 63 : index> : index
  %bound = test.with_bounds <smin = 16 : index, smax = 112 : index, umin = 16 : index, umax = 112 : index> : index
  scf.for %arg0 = %c16 to %bound step %c64 {
    %0 = arith.subi %bound, %arg0 : index
    %1 = arith.minsi %0, %c64 : index
    %2 = arith.subi %1, %tid : index
    %3 = arith.cmpi slt, %2, %c0 : index
    scf.if %3 {
      vector.print str "sideeffect"
    }
  }
  return
}

func.func private @use(index)
func.func private @use_i64(i64)

// CHECK-LABEL: func.func @loop_with_iter_arg
func.func @loop_with_iter_arg() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index

  %cst = arith.constant dense<0.000000e+00> : vector<4xf32>

//       CHECK:  %[[POS:.*]] = test.with_bounds <umin = 0 : index, umax = 1 : index, smin = 0 : index, smax = 1 : index> : index
//       CHECK:  %[[NEG:.*]] = test.with_bounds <umin = 0 : index, umax = -1 : index, smin = -1 : index, smax = 0 : index> : index
// Check iter args are still present
//       CHECK:  scf.for {{.*}} iter_args({{.*}})
//       CHECK:  %[[POS_I8:.*]] = arith.index_castui %[[POS]] : index to i8
//       CHECK:  %[[NEG_I8:.*]] = arith.index_cast %[[NEG]] : index to i8
//       CHECK:  %[[RES_I8:.*]] = arith.addi %[[POS_I8]], %[[NEG_I8]] : i8
//       CHECK:  %[[RES:.*]] = arith.index_cast %[[RES_I8]] : i8 to index
//       CHECK:  call @use(%[[RES]])

  %0 = test.with_bounds < umin = 0 : index, umax = 1 : index, smin = 0 : index, smax = 1 : index > : index
  %1 = test.with_bounds < umin = 0 : index, umax = -1 : index, smin = -1 : index, smax = 0 : index > : index
  %res = scf.for %arg0 = %c0 to %c16 step %c1 iter_args(%arg1 = %cst) -> (vector<4xf32>) {
    %2 = arith.addi %0, %1 : index
    func.call @use(%2) : (index) -> ()
    scf.yield %arg1 : vector<4xf32>
  }
  return
}

//===----------------------------------------------------------------------===//
// Loop bounds narrowing
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @narrow_loop_bounds
func.func @narrow_loop_bounds() {
  %c0_i64 = arith.constant 0 : i64
  %c10_i64 = arith.constant 10 : i64
  %c1_i64 = arith.constant 1 : i64

  // CHECK-DAG: %[[C1_I8:.*]] = arith.constant 1 : i8
  // CHECK-DAG: %[[LB:.*]] = test.with_bounds <umin = 0 : i64, umax = 0 : i64, smin = 0 : i64, smax = 0 : i64> : i64
  // CHECK-DAG: %[[UB:.*]] = test.with_bounds <umin = 10 : i64, umax = 10 : i64, smin = 10 : i64, smax = 10 : i64> : i64
  // CHECK-DAG: %[[STEP:.*]] = test.with_bounds <umin = 1 : i64, umax = 1 : i64, smin = 1 : i64, smax = 1 : i64> : i64
  %lb = test.with_bounds <smin = 0 : i64, smax = 0 : i64, umin = 0 : i64, umax = 0 : i64> : i64
  %ub = test.with_bounds <smin = 10 : i64, smax = 10 : i64, umin = 10 : i64, umax = 10 : i64> : i64
  %step = test.with_bounds <smin = 1 : i64, smax = 1 : i64, umin = 1 : i64, umax = 1 : i64> : i64

  // CHECK-DAG: %[[LB_I8:.*]] = arith.trunci %[[LB]] : i64 to i8
  // CHECK-DAG: %[[UB_I8:.*]] = arith.trunci %[[UB]] : i64 to i8
  // CHECK-DAG: %[[STEP_I8:.*]] = arith.trunci %[[STEP]] : i64 to i8
  // CHECK: scf.for %[[IV:.*]] = %[[LB_I8]] to %[[UB_I8]] step %[[STEP_I8]] : i8 {
  // CHECK:   %[[ADD_I8:.*]] = arith.addi %[[IV]], %[[C1_I8]] : i8
  // CHECK:   %[[ADD_I64:.*]] = arith.extui %[[ADD_I8]] : i8 to i64
  // CHECK:   call @use_i64(%[[ADD_I64]])
  scf.for %iv = %lb to %ub step %step : i64 {
    %add = arith.addi %iv, %c1_i64 : i64
    func.call @use_i64(%add) : (i64) -> ()
  }
  return
}

// Ensure affine.for with a non-constant (dynamic) upper bound doesn't crash.
// affine.for implements LoopLikeOpInterface but getLoopUpperBounds() returns
// nullopt for non-constant bounds, which must be handled gracefully.
func.func private @use_index(index) -> ()
// CHECK-LABEL: func @affine_for_dynamic_bound
func.func @affine_for_dynamic_bound(%n: index) {
  // CHECK: affine.for %[[IV:.*]] = 0 to %{{.*}} {
  affine.for %i = 0 to %n {
    // CHECK: func.call @use_index(%[[IV]])
    func.call @use_index(%i) : (index) -> ()
  }
  return
}

// Ensure a zero-element constant tensor doesn't crash range inference.
// CHECK-LABEL: func @zero_element_tensor_constant
func.func @zero_element_tensor_constant() -> tensor<2x0xi32> {
  // CHECK: arith.constant dense<> : tensor<2x0xi32>
  %0 = arith.constant dense<> : tensor<2x0xi32>
  return %0 : tensor<2x0xi32>
}

// Narrowing introduces the exceptional pair at i8. Preserve the original
// defined zero result even though the original operation has no flag.
// CHECK-LABEL: @remsi_introduced_overflow
// CHECK: arith.remsi {{.*}} overflow<zero> : i8
// CHECK: arith.extsi {{.*}} : i8 to i32
func.func @remsi_introduced_overflow() -> i32 {
  %a = test.with_bounds <smin = -128 : i32, smax = -127 : i32, umin = -128 : i32, umax = -127 : i32> : i32
  %b = test.with_bounds <smin = -2 : i32, smax = -1 : i32, umin = -2 : i32, umax = -1 : i32> : i32
  %r = arith.remsi %a, %b : i32
  return %r : i32
}

// CHECK-LABEL: @remsi_no_introduced_overflow
// CHECK: arith.remsi %{{[^,]+}}, %{{[^ ]+}} : i8
// CHECK: return
func.func @remsi_no_introduced_overflow() -> i32 {
  %a = test.with_bounds <smin = -127 : i32, smax = -126 : i32, umin = -127 : i32, umax = -126 : i32> : i32
  %b = test.with_bounds <smin = -2 : i32, smax = -1 : i32, umin = -2 : i32, umax = -1 : i32> : i32
  %r = arith.remsi %a, %b : i32
  return %r : i32
}

// CHECK-LABEL: @remsi_preserve_overflow_zero
// CHECK: arith.remsi {{.*}} overflow<zero> : i8
// CHECK: arith.extsi {{.*}} : i8 to i32
func.func @remsi_preserve_overflow_zero() -> i32 {
  %a = test.with_bounds <smin = -128 : i32, smax = -127 : i32, umin = -128 : i32, umax = -127 : i32> : i32
  %b = test.with_bounds <smin = -2 : i32, smax = -1 : i32, umin = -2 : i32, umax = -1 : i32> : i32
  %r = arith.remsi %a, %b overflow<zero> : i32
  return %r : i32
}
