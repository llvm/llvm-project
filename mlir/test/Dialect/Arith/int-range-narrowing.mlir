// RUN: mlir-opt --arith-int-range-narrowing="int-bitwidths-supported=1,8,16,24,32" %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Some basic tests
//===----------------------------------------------------------------------===//

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
//       CHECK:  %[[A_CASTED:.*]] = arith.index_castui %[[A]] : index to i8
//       CHECK:  %[[B_CASTED:.*]] = arith.index_castui %[[B]] : index to i8
//       CHECK:  %[[RES:.*]] = arith.addi %[[A_CASTED]], %[[B_CASTED]] : i8
//       CHECK:  %[[RES_CASTED:.*]] = arith.index_castui %[[RES]] : i8 to index
//       CHECK:  return %[[RES_CASTED]] : index
func.func @test_addi() -> index {
  %0 = test.with_bounds { umin = 4 : index, umax = 5 : index, smin = 4 : index, smax = 5 : index } : index
  %1 = test.with_bounds { umin = 6 : index, umax = 7 : index, smin = 6 : index, smax = 7 : index } : index
  %2 = arith.addi %0, %1 : index
  return %2 : index
}

// CHECK-LABEL: func @test_addi_vec
//       CHECK:  %[[A:.*]] = test.with_bounds {smax = 5 : index, smin = 4 : index, umax = 5 : index, umin = 4 : index} : vector<4xindex>
//       CHECK:  %[[B:.*]] = test.with_bounds {smax = 7 : index, smin = 6 : index, umax = 7 : index, umin = 6 : index} : vector<4xindex>
//       CHECK:  %[[A_CASTED:.*]] = arith.index_castui %[[A]] : vector<4xindex> to vector<4xi8>
//       CHECK:  %[[B_CASTED:.*]] = arith.index_castui %[[B]] : vector<4xindex> to vector<4xi8>
//       CHECK:  %[[RES:.*]] = arith.addi %[[A_CASTED]], %[[B_CASTED]] : vector<4xi8>
//       CHECK:  %[[RES_CASTED:.*]] = arith.index_castui %[[RES]] : vector<4xi8> to vector<4xindex>
//       CHECK:  return %[[RES_CASTED]] : vector<4xindex>
func.func @test_addi_vec() -> vector<4xindex> {
  %0 = test.with_bounds { umin = 4 : index, umax = 5 : index, smin = 4 : index, smax = 5 : index } : vector<4xindex>
  %1 = test.with_bounds { umin = 6 : index, umax = 7 : index, smin = 6 : index, smax = 7 : index } : vector<4xindex>
  %2 = arith.addi %0, %1 : vector<4xindex>
  return %2 : vector<4xindex>
}

// CHECK-LABEL: func @test_addi_i64
//       CHECK:  %[[A:.*]] = test.with_bounds {smax = 5 : i64, smin = 4 : i64, umax = 5 : i64, umin = 4 : i64} : i64
//       CHECK:  %[[B:.*]] = test.with_bounds {smax = 7 : i64, smin = 6 : i64, umax = 7 : i64, umin = 6 : i64} : i64
//       CHECK:  %[[A_CASTED:.*]] = arith.trunci %[[A]] : i64 to i8
//       CHECK:  %[[B_CASTED:.*]] = arith.trunci %[[B]] : i64 to i8
//       CHECK:  %[[RES:.*]] = arith.addi %[[A_CASTED]], %[[B_CASTED]] : i8
//       CHECK:  %[[RES_CASTED:.*]] = arith.extui %[[RES]] : i8 to i64
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
//       CHECK:  %[[A_CASTED:.*]] = arith.index_castui %[[A]] : index to i8
//       CHECK:  %[[B_CASTED:.*]] = arith.index_castui %[[B]] : index to i8
//       CHECK:  %[[RES:.*]] = arith.cmpi slt, %[[A_CASTED]], %[[B_CASTED]] : i8
//       CHECK:  return %[[RES]] : i1
func.func @test_cmpi() -> i1 {
  %0 = test.with_bounds { umin = 0 : index, umax = 10 : index, smin = 0 : index, smax = 10 : index } : index
  %1 = test.with_bounds { umin = 0 : index, umax = 10 : index, smin = 0 : index, smax = 10 : index } : index
  %2 = arith.cmpi slt, %0, %1 : index
  return %2 : i1
}

// CHECK-LABEL: func @test_cmpi_vec
//       CHECK:  %[[A:.*]] = test.with_bounds {smax = 10 : index, smin = 0 : index, umax = 10 : index, umin = 0 : index} : vector<4xindex>
//       CHECK:  %[[B:.*]] = test.with_bounds {smax = 10 : index, smin = 0 : index, umax = 10 : index, umin = 0 : index} : vector<4xindex>
//       CHECK:  %[[A_CASTED:.*]] = arith.index_castui %[[A]] : vector<4xindex> to vector<4xi8>
//       CHECK:  %[[B_CASTED:.*]] = arith.index_castui %[[B]] : vector<4xindex> to vector<4xi8>
//       CHECK:  %[[RES:.*]] = arith.cmpi slt, %[[A_CASTED]], %[[B_CASTED]] : vector<4xi8>
//       CHECK:  return %[[RES]] : vector<4xi1>
func.func @test_cmpi_vec() -> vector<4xi1> {
  %0 = test.with_bounds { umin = 0 : index, umax = 10 : index, smin = 0 : index, smax = 10 : index } : vector<4xindex>
  %1 = test.with_bounds { umin = 0 : index, umax = 10 : index, smin = 0 : index, smax = 10 : index } : vector<4xindex>
  %2 = arith.cmpi slt, %0, %1 : vector<4xindex>
  return %2 : vector<4xi1>
}

// CHECK-LABEL: func @test_add_cmpi
//       CHECK:  %[[A:.*]] = test.with_bounds {smax = 10 : index, smin = 0 : index, umax = 10 : index, umin = 0 : index} : index
//       CHECK:  %[[B:.*]] = test.with_bounds {smax = 10 : index, smin = 0 : index, umax = 10 : index, umin = 0 : index} : index
//       CHECK:  %[[C:.*]] = test.with_bounds {smax = 10 : index, smin = 0 : index, umax = 10 : index, umin = 0 : index} : index
//       CHECK:  %[[A_CASTED:.*]] = arith.index_castui %[[A]] : index to i8
//       CHECK:  %[[B_CASTED:.*]] = arith.index_castui %[[B]] : index to i8
//       CHECK:  %[[RES1:.*]] = arith.addi %[[A_CASTED]], %[[B_CASTED]] : i8
//       CHECK:  %[[C_CASTED:.*]] = arith.index_castui %[[C]] : index to i8
//       CHECK:  %[[RES2:.*]] = arith.cmpi slt, %[[C_CASTED]], %[[RES1]] : i8
//       CHECK:  return %[[RES2]] : i1
func.func @test_add_cmpi() -> i1 {
  %0 = test.with_bounds { umin = 0 : index, umax = 10 : index, smin = 0 : index, smax = 10 : index } : index
  %1 = test.with_bounds { umin = 0 : index, umax = 10 : index, smin = 0 : index, smax = 10 : index } : index
  %3 = test.with_bounds { umin = 0 : index, umax = 10 : index, smin = 0 : index, smax = 10 : index } : index
  %4 = arith.addi %0, %1 : index
  %5 = arith.cmpi slt, %3, %4 : index
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

// This case should not get optimized because of mixed extensions.
//
// CHECK-LABEL: func.func @addi_mixed_ext_i8
// CHECK-SAME:    (%[[ARG0:.+]]: i8, %[[ARG1:.+]]: i8)
// CHECK-NEXT:    %[[EXT0:.+]] = arith.extsi %[[ARG0]] : i8 to i32
// CHECK-NEXT:    %[[EXT1:.+]] = arith.extui %[[ARG1]] : i8 to i32
// CHECK-NEXT:    %[[ADD:.+]]  = arith.addi %[[EXT0]], %[[EXT1]] : i32
// CHECK-NEXT:    return %[[ADD]] : i32
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

// This patterns should only apply to `arith.subi` ops with sign-extended
// arguments.
//
// CHECK-LABEL: func.func @subi_extui_i8
// CHECK-SAME:    (%[[ARG0:.+]]: i8, %[[ARG1:.+]]: i8)
// CHECK-NEXT:    %[[EXT0:.+]] = arith.extui %[[ARG0]] : i8 to i32
// CHECK-NEXT:    %[[EXT1:.+]] = arith.extui %[[ARG1]] : i8 to i32
// CHECK-NEXT:    %[[SUB:.+]]  = arith.subi %[[EXT0]], %[[EXT1]] : i32
// CHECK-NEXT:    return %[[SUB]] : i32
func.func @subi_extui_i8(%lhs: i8, %rhs: i8) -> i32 {
  %a = arith.extui %lhs : i8 to i32
  %b = arith.extui %rhs : i8 to i32
  %r = arith.subi %a, %b : i32
  return %r : i32
}

// This case should not get optimized because of mixed extensions.
//
// CHECK-LABEL: func.func @subi_mixed_ext_i8
// CHECK-SAME:    (%[[ARG0:.+]]: i8, %[[ARG1:.+]]: i8)
// CHECK-NEXT:    %[[EXT0:.+]] = arith.extsi %[[ARG0]] : i8 to i32
// CHECK-NEXT:    %[[EXT1:.+]] = arith.extui %[[ARG1]] : i8 to i32
// CHECK-NEXT:    %[[ADD:.+]]  = arith.subi %[[EXT0]], %[[EXT1]] : i32
// CHECK-NEXT:    return %[[ADD]] : i32
func.func @subi_mixed_ext_i8(%lhs: i8, %rhs: i8) -> i32 {
  %a = arith.extsi %lhs : i8 to i32
  %b = arith.extui %rhs : i8 to i32
  %r = arith.subi %a, %b : i32
  return %r : i32
}

//===----------------------------------------------------------------------===//
// arith.muli
//===----------------------------------------------------------------------===//

// TODO: This should be optimized into i16
// CHECK-LABEL: func.func @muli_extui_i8
// CHECK-SAME:    (%[[ARG0:.+]]: i8, %[[ARG1:.+]]: i8)
// CHECK-NEXT:    %[[EXT0:.+]] = arith.extui %[[ARG0]] : i8 to i32
// CHECK-NEXT:    %[[EXT1:.+]] = arith.extui %[[ARG1]] : i8 to i32
// CHECK-NEXT:    %[[LHS:.+]]  = arith.trunci %[[EXT0]] : i32 to i24
// CHECK-NEXT:    %[[RHS:.+]]  = arith.trunci %[[EXT1]] : i32 to i24
// CHECK-NEXT:    %[[MUL:.+]]  = arith.muli %[[LHS]], %[[RHS]] : i24
// CHECK-NEXT:    %[[RET:.+]]  = arith.extui %[[MUL]] : i24 to i32
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

// This case should not get optimized because of mixed extensions.
//
// CHECK-LABEL: func.func @muli_mixed_ext_i8
// CHECK-SAME:    (%[[ARG0:.+]]: i8, %[[ARG1:.+]]: i8)
// CHECK-NEXT:    %[[EXT0:.+]] = arith.extsi %[[ARG0]] : i8 to i32
// CHECK-NEXT:    %[[EXT1:.+]] = arith.extui %[[ARG1]] : i8 to i32
// CHECK-NEXT:    %[[MUL:.+]]  = arith.muli %[[EXT0]], %[[EXT1]] : i32
// CHECK-NEXT:    return %[[MUL]] : i32
func.func @muli_mixed_ext_i8(%lhs: i8, %rhs: i8) -> i32 {
  %a = arith.extsi %lhs : i8 to i32
  %b = arith.extui %rhs : i8 to i32
  %r = arith.muli %a, %b : i32
  return %r : i32
}
