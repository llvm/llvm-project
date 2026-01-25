// RUN: mlir-opt --arith-int-range-narrowing="int-bitwidths-supported=1,8,16,24,32" %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Some basic tests
//===----------------------------------------------------------------------===//

// Truncate possibly-negative values in a signed way
// CHECK-LABEL: func @test_addi_neg
//       CHECK:  %[[POS:.*]] = test.with_bounds {smax = 1 : index, smin = 0 : index, umax = 1 : index, umin = 0 : index} : index
//       CHECK:  %[[NEG:.*]] = test.with_bounds {smax = 0 : index, smin = -1 : index, umax = -1 : index, umin = 0 : index} : index
//       CHECK:  %[[POS_I8:.*]] = arith.index_castui %[[POS]] : index to i8
//       CHECK:  %[[NEG_I8:.*]] = arith.index_cast %[[NEG]] : index to i8
//       CHECK:  %[[RES_I8:.*]] = arith.addi %[[POS_I8]], %[[NEG_I8]] : i8
//       CHECK:  %[[RES:.*]] = arith.index_cast %[[RES_I8]] : i8 to index
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

// CHECK-LABEL: func @test_add_cmpi_i64
//       CHECK:  %[[A:.*]] = test.with_bounds {smax = 10 : i64, smin = 0 : i64, umax = 10 : i64, umin = 0 : i64} : i64
//       CHECK:  %[[B:.*]] = test.with_bounds {smax = 10 : i64, smin = 0 : i64, umax = 10 : i64, umin = 0 : i64} : i64
//       CHECK:  %[[C:.*]] = test.with_bounds {smax = 10 : i64, smin = 0 : i64, umax = 10 : i64, umin = 0 : i64} : i64
//       CHECK:  %[[A_CASTED:.*]] = arith.trunci %[[A]] : i64 to i8
//       CHECK:  %[[B_CASTED:.*]] = arith.trunci %[[B]] : i64 to i8
//       CHECK:  %[[RES1:.*]] = arith.addi %[[A_CASTED]], %[[B_CASTED]] : i8
//       CHECK:  %[[C_CASTED:.*]] = arith.trunci %[[C]] : i64 to i8
//       CHECK:  %[[RES2:.*]] = arith.cmpi slt, %[[C_CASTED]], %[[RES1]] : i8
//       CHECK:  return %[[RES2]] : i1
func.func @test_add_cmpi_i64() -> i1 {
  %0 = test.with_bounds { umin = 0 : i64, umax = 10 : i64, smin = 0 : i64, smax = 10 : i64 } : i64
  %1 = test.with_bounds { umin = 0 : i64, umax = 10 : i64, smin = 0 : i64, smax = 10 : i64 } : i64
  %3 = test.with_bounds { umin = 0 : i64, umax = 10 : i64, smin = 0 : i64, smax = 10 : i64 } : i64
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

  %tid = test.with_bounds {smin = 0 : index, smax = 63 : index, umin = 0 : index, umax = 63 : index} : index
  %bound = test.with_bounds {smin = 16 : index, smax = 112 : index, umin = 16 : index, umax = 112 : index} : index
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
  %c1 = arith.constant 0 : index
  %c16 = arith.constant 16 : index

  %cst = arith.constant dense<0.000000e+00> : vector<4xf32>

//       CHECK:  %[[POS:.*]] = test.with_bounds {smax = 1 : index, smin = 0 : index, umax = 1 : index, umin = 0 : index} : index
//       CHECK:  %[[NEG:.*]] = test.with_bounds {smax = 0 : index, smin = -1 : index, umax = -1 : index, umin = 0 : index} : index
// Check iter args are still present
//       CHECK:  scf.for {{.*}} iter_args({{.*}})
//       CHECK:  %[[POS_I8:.*]] = arith.index_castui %[[POS]] : index to i8
//       CHECK:  %[[NEG_I8:.*]] = arith.index_cast %[[NEG]] : index to i8
//       CHECK:  %[[RES_I8:.*]] = arith.addi %[[POS_I8]], %[[NEG_I8]] : i8
//       CHECK:  %[[RES:.*]] = arith.index_cast %[[RES_I8]] : i8 to index
//       CHECK:  call @use(%[[RES]])

  %0 = test.with_bounds { umin = 0 : index, umax = 1 : index, smin = 0 : index, smax = 1 : index } : index
  %1 = test.with_bounds { umin = 0 : index, umax = -1 : index, smin = -1 : index, smax = 0 : index } : index
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
  // CHECK-DAG: %[[LB:.*]] = test.with_bounds {smax = 0 : i64, smin = 0 : i64, umax = 0 : i64, umin = 0 : i64} : i64
  // CHECK-DAG: %[[UB:.*]] = test.with_bounds {smax = 10 : i64, smin = 10 : i64, umax = 10 : i64, umin = 10 : i64} : i64
  // CHECK-DAG: %[[STEP:.*]] = test.with_bounds {smax = 1 : i64, smin = 1 : i64, umax = 1 : i64, umin = 1 : i64} : i64
  %lb = test.with_bounds {smin = 0 : i64, smax = 0 : i64, umin = 0 : i64, umax = 0 : i64} : i64
  %ub = test.with_bounds {smin = 10 : i64, smax = 10 : i64, umin = 10 : i64, umax = 10 : i64} : i64
  %step = test.with_bounds {smin = 1 : i64, smax = 1 : i64, umin = 1 : i64, umax = 1 : i64} : i64

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
