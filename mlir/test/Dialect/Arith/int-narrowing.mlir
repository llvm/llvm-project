// RUN: mlir-opt --arith-int-narrowing="int-bitwidths-supported=1,8,16,24,32" \
// RUN:          --verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// arith.addi
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @addi_extsi_i8
// CHECK-SAME:    (%[[ARG0:.+]]: i8, %[[ARG1:.+]]: i8)
// CHECK-NEXT:    %[[EXT0:.+]] = arith.extsi %[[ARG0]] : i8 to i32
// CHECK-NEXT:    %[[EXT1:.+]] = arith.extsi %[[ARG1]] : i8 to i32
// CHECK-NEXT:    %[[LHS:.+]]  = arith.trunci %[[EXT0]] : i32 to i16
// CHECK-NEXT:    %[[RHS:.+]]  = arith.trunci %[[EXT1]] : i32 to i16
// CHECK-NEXT:    %[[ADD:.+]]  = arith.addi %[[LHS]], %[[RHS]] : i16
// CHECK-NEXT:    %[[RET:.+]]  = arith.extsi %[[ADD]] : i16 to i32
// CHECK-NEXT:    return %[[RET]] : i32
func.func @addi_extsi_i8(%lhs: i8, %rhs: i8) -> i32 {
  %a = arith.extsi %lhs : i8 to i32
  %b = arith.extsi %rhs : i8 to i32
  %r = arith.addi %a, %b : i32
  return %r : i32
}

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

// arith.addi produces one more bit of result than the operand bitwidth.
//
// CHECK-LABEL: func.func @addi_extsi_i24
// CHECK-SAME:    (%[[ARG0:.+]]: i16, %[[ARG1:.+]]: i16)
// CHECK-NEXT:    %[[EXT0:.+]] = arith.extsi %[[ARG0]] : i16 to i32
// CHECK-NEXT:    %[[EXT1:.+]] = arith.extsi %[[ARG1]] : i16 to i32
// CHECK-NEXT:    %[[LHS:.+]]  = arith.trunci %[[EXT0]] : i32 to i24
// CHECK-NEXT:    %[[RHS:.+]]  = arith.trunci %[[EXT1]] : i32 to i24
// CHECK-NEXT:    %[[ADD:.+]]  = arith.addi %[[LHS]], %[[RHS]] : i24
// CHECK-NEXT:    %[[RET:.+]]  = arith.extsi %[[ADD]] : i24 to i32
// CHECK-NEXT:    return %[[RET]] : i32
func.func @addi_extsi_i24(%lhs: i16, %rhs: i16) -> i32 {
  %a = arith.extsi %lhs : i16 to i32
  %b = arith.extsi %rhs : i16 to i32
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

// CHECK-LABEL: func.func @addi_extsi_3xi8_cst
// CHECK-SAME:    (%[[ARG0:.+]]: vector<3xi8>)
// CHECK-NEXT:    %[[CST:.+]]  = arith.constant dense<[-1, 127, 42]> : vector<3xi16>
// CHECK-NEXT:    %[[EXT:.+]]  = arith.extsi %[[ARG0]] : vector<3xi8> to vector<3xi32>
// CHECK-NEXT:    %[[LHS:.+]]  = arith.trunci %[[EXT]] : vector<3xi32> to vector<3xi16>
// CHECK-NEXT:    %[[ADD:.+]]  = arith.addi %[[LHS]], %[[CST]] : vector<3xi16>
// CHECK-NEXT:    %[[RET:.+]]  = arith.extsi %[[ADD]] : vector<3xi16> to vector<3xi32>
// CHECK-NEXT:    return %[[RET]] : vector<3xi32>
func.func @addi_extsi_3xi8_cst(%lhs: vector<3xi8>) -> vector<3xi32> {
  %cst = arith.constant dense<[-1, 127, 42]> : vector<3xi32>
  %a = arith.extsi %lhs : vector<3xi8> to vector<3xi32>
  %r = arith.addi %a, %cst : vector<3xi32>
  return %r : vector<3xi32>
}

//===----------------------------------------------------------------------===//
// arith.subi
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @subi_extsi_i8
// CHECK-SAME:    (%[[ARG0:.+]]: i8, %[[ARG1:.+]]: i8)
// CHECK-NEXT:    %[[EXT0:.+]] = arith.extsi %[[ARG0]] : i8 to i32
// CHECK-NEXT:    %[[EXT1:.+]] = arith.extsi %[[ARG1]] : i8 to i32
// CHECK-NEXT:    %[[LHS:.+]]  = arith.trunci %[[EXT0]] : i32 to i16
// CHECK-NEXT:    %[[RHS:.+]]  = arith.trunci %[[EXT1]] : i32 to i16
// CHECK-NEXT:    %[[SUB:.+]]  = arith.subi %[[LHS]], %[[RHS]] : i16
// CHECK-NEXT:    %[[RET:.+]]  = arith.extsi %[[SUB]] : i16 to i32
// CHECK-NEXT:    return %[[RET]] : i32
func.func @subi_extsi_i8(%lhs: i8, %rhs: i8) -> i32 {
  %a = arith.extsi %lhs : i8 to i32
  %b = arith.extsi %rhs : i8 to i32
  %r = arith.subi %a, %b : i32
  return %r : i32
}

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

// arith.subi produces one more bit of result than the operand bitwidth.
//
// CHECK-LABEL: func.func @subi_extsi_i24
// CHECK-SAME:    (%[[ARG0:.+]]: i16, %[[ARG1:.+]]: i16)
// CHECK-NEXT:    %[[EXT0:.+]] = arith.extsi %[[ARG0]] : i16 to i32
// CHECK-NEXT:    %[[EXT1:.+]] = arith.extsi %[[ARG1]] : i16 to i32
// CHECK-NEXT:    %[[LHS:.+]]  = arith.trunci %[[EXT0]] : i32 to i24
// CHECK-NEXT:    %[[RHS:.+]]  = arith.trunci %[[EXT1]] : i32 to i24
// CHECK-NEXT:    %[[ADD:.+]]  = arith.subi %[[LHS]], %[[RHS]] : i24
// CHECK-NEXT:    %[[RET:.+]]  = arith.extsi %[[ADD]] : i24 to i32
// CHECK-NEXT:    return %[[RET]] : i32
func.func @subi_extsi_i24(%lhs: i16, %rhs: i16) -> i32 {
  %a = arith.extsi %lhs : i16 to i32
  %b = arith.extsi %rhs : i16 to i32
  %r = arith.subi %a, %b : i32
  return %r : i32
}

//===----------------------------------------------------------------------===//
// arith.muli
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @muli_extsi_i8
// CHECK-SAME:    (%[[ARG0:.+]]: i8, %[[ARG1:.+]]: i8)
// CHECK-NEXT:    %[[EXT0:.+]] = arith.extsi %[[ARG0]] : i8 to i32
// CHECK-NEXT:    %[[EXT1:.+]] = arith.extsi %[[ARG1]] : i8 to i32
// CHECK-NEXT:    %[[LHS:.+]]  = arith.trunci %[[EXT0]] : i32 to i16
// CHECK-NEXT:    %[[RHS:.+]]  = arith.trunci %[[EXT1]] : i32 to i16
// CHECK-NEXT:    %[[MUL:.+]]  = arith.muli %[[LHS]], %[[RHS]] : i16
// CHECK-NEXT:    %[[RET:.+]]  = arith.extsi %[[MUL]] : i16 to i32
// CHECK-NEXT:    return %[[RET]] : i32
func.func @muli_extsi_i8(%lhs: i8, %rhs: i8) -> i32 {
  %a = arith.extsi %lhs : i8 to i32
  %b = arith.extsi %rhs : i8 to i32
  %r = arith.muli %a, %b : i32
  return %r : i32
}

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

// CHECK-LABEL: func.func @muli_extsi_3xi8_cst
// CHECK-SAME:    (%[[ARG0:.+]]: vector<3xi8>)
// CHECK-NEXT:    %[[CST:.+]]  = arith.constant dense<[-1, 127, 42]> : vector<3xi16>
// CHECK-NEXT:    %[[EXT:.+]]  = arith.extsi %[[ARG0]] : vector<3xi8> to vector<3xi32>
// CHECK-NEXT:    %[[LHS:.+]]  = arith.trunci %[[EXT]] : vector<3xi32> to vector<3xi16>
// CHECK-NEXT:    %[[MUL:.+]]  = arith.muli %[[LHS]], %[[CST]] : vector<3xi16>
// CHECK-NEXT:    %[[RET:.+]]  = arith.extsi %[[MUL]] : vector<3xi16> to vector<3xi32>
// CHECK-NEXT:    return %[[RET]] : vector<3xi32>
func.func @muli_extsi_3xi8_cst(%lhs: vector<3xi8>) -> vector<3xi32> {
  %cst = arith.constant dense<[-1, 127, 42]> : vector<3xi32>
  %a = arith.extsi %lhs : vector<3xi8> to vector<3xi32>
  %r = arith.muli %a, %cst : vector<3xi32>
  return %r : vector<3xi32>
}

//===----------------------------------------------------------------------===//
// arith.divsi
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @divsi_extsi_i8
// CHECK-SAME:    (%[[ARG0:.+]]: i8, %[[ARG1:.+]]: i8)
// CHECK-NEXT:    %[[EXT0:.+]] = arith.extsi %[[ARG0]] : i8 to i32
// CHECK-NEXT:    %[[EXT1:.+]] = arith.extsi %[[ARG1]] : i8 to i32
// CHECK-NEXT:    %[[LHS:.+]]  = arith.trunci %[[EXT0]] : i32 to i16
// CHECK-NEXT:    %[[RHS:.+]]  = arith.trunci %[[EXT1]] : i32 to i16
// CHECK-NEXT:    %[[SUB:.+]]  = arith.divsi %[[LHS]], %[[RHS]] : i16
// CHECK-NEXT:    %[[RET:.+]]  = arith.extsi %[[SUB]] : i16 to i32
// CHECK-NEXT:    return %[[RET]] : i32
func.func @divsi_extsi_i8(%lhs: i8, %rhs: i8) -> i32 {
  %a = arith.extsi %lhs : i8 to i32
  %b = arith.extsi %rhs : i8 to i32
  %r = arith.divsi %a, %b : i32
  return %r : i32
}

// This patterns should only apply to `arith.divsi` ops with sign-extended
// arguments.
//
// CHECK-LABEL: func.func @divsi_extui_i8
// CHECK-SAME:    (%[[ARG0:.+]]: i8, %[[ARG1:.+]]: i8)
// CHECK-NEXT:    %[[EXT0:.+]] = arith.extui %[[ARG0]] : i8 to i32
// CHECK-NEXT:    %[[EXT1:.+]] = arith.extui %[[ARG1]] : i8 to i32
// CHECK-NEXT:    %[[SUB:.+]]  = arith.divsi %[[EXT0]], %[[EXT1]] : i32
// CHECK-NEXT:    return %[[SUB]] : i32
func.func @divsi_extui_i8(%lhs: i8, %rhs: i8) -> i32 {
  %a = arith.extui %lhs : i8 to i32
  %b = arith.extui %rhs : i8 to i32
  %r = arith.divsi %a, %b : i32
  return %r : i32
}

// arith.divsi produces one more bit of result than the operand bitwidth.
//
// CHECK-LABEL: func.func @divsi_extsi_i24
// CHECK-SAME:    (%[[ARG0:.+]]: i16, %[[ARG1:.+]]: i16)
// CHECK-NEXT:    %[[EXT0:.+]] = arith.extsi %[[ARG0]] : i16 to i32
// CHECK-NEXT:    %[[EXT1:.+]] = arith.extsi %[[ARG1]] : i16 to i32
// CHECK-NEXT:    %[[LHS:.+]]  = arith.trunci %[[EXT0]] : i32 to i24
// CHECK-NEXT:    %[[RHS:.+]]  = arith.trunci %[[EXT1]] : i32 to i24
// CHECK-NEXT:    %[[ADD:.+]]  = arith.divsi %[[LHS]], %[[RHS]] : i24
// CHECK-NEXT:    %[[RET:.+]]  = arith.extsi %[[ADD]] : i24 to i32
// CHECK-NEXT:    return %[[RET]] : i32
func.func @divsi_extsi_i24(%lhs: i16, %rhs: i16) -> i32 {
  %a = arith.extsi %lhs : i16 to i32
  %b = arith.extsi %rhs : i16 to i32
  %r = arith.divsi %a, %b : i32
  return %r : i32
}

//===----------------------------------------------------------------------===//
// arith.divui
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @divui_extui_i8
// CHECK-SAME:    (%[[ARG0:.+]]: i8, %[[ARG1:.+]]: i8)
// CHECK-NEXT:    %[[SUB:.+]]  = arith.divui %[[ARG0]], %[[ARG1]] : i8
// CHECK-NEXT:    %[[RET:.+]]  = arith.extui %[[SUB]] : i8 to i32
// CHECK-NEXT:    return %[[RET]] : i32
func.func @divui_extui_i8(%lhs: i8, %rhs: i8) -> i32 {
  %a = arith.extui %lhs : i8 to i32
  %b = arith.extui %rhs : i8 to i32
  %r = arith.divui %a, %b : i32
  return %r : i32
}

// This patterns should only apply to `arith.divui` ops with zero-extended
// arguments.
//
// CHECK-LABEL: func.func @divui_extsi_i8
// CHECK-SAME:    (%[[ARG0:.+]]: i8, %[[ARG1:.+]]: i8)
// CHECK-NEXT:    %[[EXT0:.+]] = arith.extsi %[[ARG0]] : i8 to i32
// CHECK-NEXT:    %[[EXT1:.+]] = arith.extsi %[[ARG1]] : i8 to i32
// CHECK-NEXT:    %[[SUB:.+]]  = arith.divui %[[EXT0]], %[[EXT1]] : i32
// CHECK-NEXT:    return %[[SUB]] : i32
func.func @divui_extsi_i8(%lhs: i8, %rhs: i8) -> i32 {
  %a = arith.extsi %lhs : i8 to i32
  %b = arith.extsi %rhs : i8 to i32
  %r = arith.divui %a, %b : i32
  return %r : i32
}

//===----------------------------------------------------------------------===//
// arith.*itofp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @sitofp_extsi_i16
// CHECK-SAME:    (%[[ARG:.+]]: i16)
// CHECK-NEXT:    %[[RET:.+]] = arith.sitofp %[[ARG]] : i16 to f16
// CHECK-NEXT:    return %[[RET]] : f16
func.func @sitofp_extsi_i16(%a: i16) -> f16 {
  %b = arith.extsi %a : i16 to i32
  %f = arith.sitofp %b : i32 to f16
  return %f : f16
}

// CHECK-LABEL: func.func @sitofp_extsi_vector_i16
// CHECK-SAME:    (%[[ARG:.+]]: vector<3xi16>)
// CHECK-NEXT:    %[[RET:.+]] = arith.sitofp %[[ARG]] : vector<3xi16> to vector<3xf16>
// CHECK-NEXT:    return %[[RET]] : vector<3xf16>
func.func @sitofp_extsi_vector_i16(%a: vector<3xi16>) -> vector<3xf16> {
  %b = arith.extsi %a : vector<3xi16> to vector<3xi32>
  %f = arith.sitofp %b : vector<3xi32> to vector<3xf16>
  return %f : vector<3xf16>
}

// CHECK-LABEL: func.func @sitofp_extsi_tensor_i16
// CHECK-SAME:    (%[[ARG:.+]]: tensor<3x?xi16>)
// CHECK-NEXT:    %[[RET:.+]] = arith.sitofp %[[ARG]] : tensor<3x?xi16> to tensor<3x?xf16>
// CHECK-NEXT:    return %[[RET]] : tensor<3x?xf16>
func.func @sitofp_extsi_tensor_i16(%a: tensor<3x?xi16>) -> tensor<3x?xf16> {
  %b = arith.extsi %a : tensor<3x?xi16> to tensor<3x?xi32>
  %f = arith.sitofp %b : tensor<3x?xi32> to tensor<3x?xf16>
  return %f : tensor<3x?xf16>
}

// Narrowing to i64 is not enabled in pass options.
//
// CHECK-LABEL: func.func @sitofp_extsi_i64
// CHECK-SAME:    (%[[ARG:.+]]: i64)
// CHECK-NEXT:    %[[EXT:.+]] = arith.extsi %[[ARG]] : i64 to i128
// CHECK-NEXT:    %[[RET:.+]] = arith.sitofp %[[EXT]] : i128 to f32
// CHECK-NEXT:    return %[[RET]] : f32
func.func @sitofp_extsi_i64(%a: i64) -> f32 {
  %b = arith.extsi %a : i64 to i128
  %f = arith.sitofp %b : i128 to f32
  return %f : f32
}

// CHECK-LABEL: func.func @uitofp_extui_i16
// CHECK-SAME:    (%[[ARG:.+]]: i16)
// CHECK-NEXT:    %[[RET:.+]] = arith.uitofp %[[ARG]] : i16 to f16
// CHECK-NEXT:    return %[[RET]] : f16
func.func @uitofp_extui_i16(%a: i16) -> f16 {
  %b = arith.extui %a : i16 to i32
  %f = arith.uitofp %b : i32 to f16
  return %f : f16
}

// CHECK-LABEL: func.func @sitofp_extsi_extsi_i8
// CHECK-SAME:    (%[[ARG:.+]]: i8)
// CHECK-NEXT:    %[[RET:.+]] = arith.sitofp %[[ARG]] : i8 to f16
// CHECK-NEXT:    return %[[RET]] : f16
func.func @sitofp_extsi_extsi_i8(%a: i8) -> f16 {
  %b = arith.extsi %a : i8 to i16
  %c = arith.extsi %b : i16 to i32
  %f = arith.sitofp %c : i32 to f16
  return %f : f16
}

// CHECK-LABEL: func.func @uitofp_extui_extui_i8
// CHECK-SAME:    (%[[ARG:.+]]: i8)
// CHECK-NEXT:    %[[RET:.+]] = arith.uitofp %[[ARG]] : i8 to f16
// CHECK-NEXT:    return %[[RET]] : f16
func.func @uitofp_extui_extui_i8(%a: i8) -> f16 {
  %b = arith.extui %a : i8 to i16
  %c = arith.extui %b : i16 to i32
  %f = arith.uitofp %c : i32 to f16
  return %f : f16
}

// CHECK-LABEL: func.func @uitofp_extsi_extui_i8
// CHECK-SAME:    (%[[ARG:.+]]: i8)
// CHECK-NEXT:    %[[EXT:.+]] = arith.extsi %[[ARG]] : i8 to i16
// CHECK-NEXT:    %[[RET:.+]] = arith.uitofp %[[EXT]] : i16 to f16
// CHECK-NEXT:    return %[[RET]] : f16
func.func @uitofp_extsi_extui_i8(%a: i8) -> f16 {
  %b = arith.extsi %a : i8 to i16
  %c = arith.extui %b : i16 to i32
  %f = arith.uitofp %c : i32 to f16
  return %f : f16
}

// CHECK-LABEL: func.func @uitofp_trunci_extui_i8
// CHECK-SAME:    (%[[ARG:.+]]: i16)
// CHECK-NEXT:    %[[TR:.+]]  = arith.trunci %[[ARG]] : i16 to i8
// CHECK-NEXT:    %[[RET:.+]] = arith.uitofp %[[TR]] : i8 to f16
// CHECK-NEXT:    return %[[RET]] : f16
func.func @uitofp_trunci_extui_i8(%a: i16) -> f16 {
  %b = arith.trunci %a : i16 to i8
  %c = arith.extui %b : i8 to i32
  %f = arith.uitofp %c : i32 to f16
  return %f : f16
}

// This should not be folded because arith.extui changes the signed
// range of the number. For example:
//  extsi -1 : i16 to i32 ==> -1
//  extui -1 : i16 to i32 ==> U16_MAX
//
/// CHECK-LABEL: func.func @sitofp_extui_i16
// CHECK-SAME:    (%[[ARG:.+]]: i16)
// CHECK-NEXT:    %[[EXT:.+]] = arith.extui %[[ARG]] : i16 to i32
// CHECK-NEXT:    %[[RET:.+]] = arith.sitofp %[[EXT]] : i32 to f16
// CHECK-NEXT:    return %[[RET]] : f16
func.func @sitofp_extui_i16(%a: i16) -> f16 {
  %b = arith.extui %a : i16 to i32
  %f = arith.sitofp %b : i32 to f16
  return %f : f16
}

// This should not be folded because arith.extsi changes the unsigned
// range of the number. For example:
//  extsi -1 : i16 to i32 ==> U32_MAX
//  extui -1 : i16 to i32 ==> U16_MAX
//
// CHECK-LABEL: func.func @uitofp_extsi_i16
// CHECK-SAME:    (%[[ARG:.+]]: i16)
// CHECK-NEXT:    %[[EXT:.+]] = arith.extsi %[[ARG]] : i16 to i32
// CHECK-NEXT:    %[[RET:.+]] = arith.uitofp %[[EXT]] : i32 to f16
// CHECK-NEXT:    return %[[RET]] : f16
func.func @uitofp_extsi_i16(%a: i16) -> f16 {
  %b = arith.extsi %a : i16 to i32
  %f = arith.uitofp %b : i32 to f16
  return %f : f16
}

//===----------------------------------------------------------------------===//
// arith.maxsi
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @maxsi_extsi_i8
// CHECK-SAME:    (%[[LHS:.+]]: i8, %[[RHS:.+]]: i8)
// CHECK-NEXT:    %[[MAX:.+]]  = arith.maxsi %[[LHS]], %[[RHS]] : i8
// CHECK-NEXT:    %[[RET:.+]]  = arith.extsi %[[MAX]] : i8 to i32
// CHECK-NEXT:    return %[[RET]] : i32
func.func @maxsi_extsi_i8(%lhs: i8, %rhs: i8) -> i32 {
  %a = arith.extsi %lhs : i8 to i32
  %b = arith.extsi %rhs : i8 to i32
  %r = arith.maxsi %a, %b : i32
  return %r : i32
}

// This patterns should only apply to `arith.maxsi` ops with sign-extended
// arguments.
//
// CHECK-LABEL: func.func @maxsi_extui_i8
// CHECK-SAME:    (%[[ARG0:.+]]: i8, %[[ARG1:.+]]: i8)
// CHECK-NEXT:    %[[EXT0:.+]] = arith.extui %[[ARG0]] : i8 to i32
// CHECK-NEXT:    %[[EXT1:.+]] = arith.extui %[[ARG1]] : i8 to i32
// CHECK-NEXT:    %[[MAX:.+]]  = arith.maxsi %[[EXT0]], %[[EXT1]] : i32
// CHECK-NEXT:    return %[[MAX]] : i32
func.func @maxsi_extui_i8(%lhs: i8, %rhs: i8) -> i32 {
  %a = arith.extui %lhs : i8 to i32
  %b = arith.extui %rhs : i8 to i32
  %r = arith.maxsi %a, %b : i32
  return %r : i32
}

//===----------------------------------------------------------------------===//
// arith.maxui
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @maxui_extui_i8
// CHECK-SAME:    (%[[LHS:.+]]: i8, %[[RHS:.+]]: i8)
// CHECK-NEXT:    %[[MAX:.+]]  = arith.maxui %[[LHS]], %[[RHS]] : i8
// CHECK-NEXT:    %[[RET:.+]]  = arith.extui %[[MAX]] : i8 to i32
// CHECK-NEXT:    return %[[RET]] : i32
func.func @maxui_extui_i8(%lhs: i8, %rhs: i8) -> i32 {
  %a = arith.extui %lhs : i8 to i32
  %b = arith.extui %rhs : i8 to i32
  %r = arith.maxui %a, %b : i32
  return %r : i32
}

// This patterns should only apply to `arith.maxsi` ops with zero-extended
// arguments.
//
// CHECK-LABEL: func.func @maxui_extsi_i8
// CHECK-SAME:    (%[[ARG0:.+]]: i8, %[[ARG1:.+]]: i8)
// CHECK-NEXT:    %[[EXT0:.+]] = arith.extsi %[[ARG0]] : i8 to i32
// CHECK-NEXT:    %[[EXT1:.+]] = arith.extsi %[[ARG1]] : i8 to i32
// CHECK-NEXT:    %[[MAX:.+]]  = arith.maxui %[[EXT0]], %[[EXT1]] : i32
// CHECK-NEXT:    return %[[MAX]] : i32
func.func @maxui_extsi_i8(%lhs: i8, %rhs: i8) -> i32 {
  %a = arith.extsi %lhs : i8 to i32
  %b = arith.extsi %rhs : i8 to i32
  %r = arith.maxui %a, %b : i32
  return %r : i32
}

//===----------------------------------------------------------------------===//
// arith.minsi
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @minsi_extsi_i8
// CHECK-SAME:    (%[[LHS:.+]]: i8, %[[RHS:.+]]: i8)
// CHECK-NEXT:    %[[min:.+]]  = arith.minsi %[[LHS]], %[[RHS]] : i8
// CHECK-NEXT:    %[[RET:.+]]  = arith.extsi %[[min]] : i8 to i32
// CHECK-NEXT:    return %[[RET]] : i32
func.func @minsi_extsi_i8(%lhs: i8, %rhs: i8) -> i32 {
  %a = arith.extsi %lhs : i8 to i32
  %b = arith.extsi %rhs : i8 to i32
  %r = arith.minsi %a, %b : i32
  return %r : i32
}

// This patterns should only apply to `arith.minsi` ops with sign-extended
// arguments.
//
// CHECK-LABEL: func.func @minsi_extui_i8
// CHECK-SAME:    (%[[ARG0:.+]]: i8, %[[ARG1:.+]]: i8)
// CHECK-NEXT:    %[[EXT0:.+]] = arith.extui %[[ARG0]] : i8 to i32
// CHECK-NEXT:    %[[EXT1:.+]] = arith.extui %[[ARG1]] : i8 to i32
// CHECK-NEXT:    %[[min:.+]]  = arith.minsi %[[EXT0]], %[[EXT1]] : i32
// CHECK-NEXT:    return %[[min]] : i32
func.func @minsi_extui_i8(%lhs: i8, %rhs: i8) -> i32 {
  %a = arith.extui %lhs : i8 to i32
  %b = arith.extui %rhs : i8 to i32
  %r = arith.minsi %a, %b : i32
  return %r : i32
}

//===----------------------------------------------------------------------===//
// arith.minui
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @minui_extui_i8
// CHECK-SAME:    (%[[LHS:.+]]: i8, %[[RHS:.+]]: i8)
// CHECK-NEXT:    %[[min:.+]]  = arith.minui %[[LHS]], %[[RHS]] : i8
// CHECK-NEXT:    %[[RET:.+]]  = arith.extui %[[min]] : i8 to i32
// CHECK-NEXT:    return %[[RET]] : i32
func.func @minui_extui_i8(%lhs: i8, %rhs: i8) -> i32 {
  %a = arith.extui %lhs : i8 to i32
  %b = arith.extui %rhs : i8 to i32
  %r = arith.minui %a, %b : i32
  return %r : i32
}

// This patterns should only apply to `arith.minsi` ops with zero-extended
// arguments.
//
// CHECK-LABEL: func.func @minui_extsi_i8
// CHECK-SAME:    (%[[ARG0:.+]]: i8, %[[ARG1:.+]]: i8)
// CHECK-NEXT:    %[[EXT0:.+]] = arith.extsi %[[ARG0]] : i8 to i32
// CHECK-NEXT:    %[[EXT1:.+]] = arith.extsi %[[ARG1]] : i8 to i32
// CHECK-NEXT:    %[[min:.+]]  = arith.minui %[[EXT0]], %[[EXT1]] : i32
// CHECK-NEXT:    return %[[min]] : i32
func.func @minui_extsi_i8(%lhs: i8, %rhs: i8) -> i32 {
  %a = arith.extsi %lhs : i8 to i32
  %b = arith.extsi %rhs : i8 to i32
  %r = arith.minui %a, %b : i32
  return %r : i32
}

//===----------------------------------------------------------------------===//
// Commute Extension over Vector Ops
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @extsi_over_extract_3xi16
// CHECK-SAME:    (%[[ARG:.+]]: vector<3xi16>)
// CHECK-NEXT:    %[[EXTR:.+]] = vector.extract %[[ARG]][1] : vector<3xi16>
// CHECK-NEXT:    %[[RET:.+]]  = arith.sitofp %[[EXTR]] : i16 to f16
// CHECK-NEXT:    return %[[RET]] : f16
func.func @extsi_over_extract_3xi16(%a: vector<3xi16>) -> f16 {
  %b = arith.extsi %a : vector<3xi16> to vector<3xi32>
  %c = vector.extract %b[1] : vector<3xi32>
  %f = arith.sitofp %c : i32 to f16
  return %f : f16
}

// CHECK-LABEL: func.func @extui_over_extract_3xi16
// CHECK-SAME:    (%[[ARG:.+]]: vector<3xi16>)
// CHECK-NEXT:    %[[EXTR:.+]] = vector.extract %[[ARG]][1] : vector<3xi16>
// CHECK-NEXT:    %[[RET:.+]]  = arith.uitofp %[[EXTR]] : i16 to f16
// CHECK-NEXT:    return %[[RET]] : f16
func.func @extui_over_extract_3xi16(%a: vector<3xi16>) -> f16 {
  %b = arith.extui %a : vector<3xi16> to vector<3xi32>
  %c = vector.extract %b[1] : vector<3xi32>
  %f = arith.uitofp %c : i32 to f16
  return %f : f16
}

// CHECK-LABEL: func.func @extsi_over_extractelement_3xi16
// CHECK-SAME:    (%[[ARG:.+]]: vector<3xi16>, %[[POS:.+]]: i32)
// CHECK-NEXT:    %[[EXTR:.+]] = vector.extractelement %[[ARG]][%[[POS]] : i32] : vector<3xi16>
// CHECK-NEXT:    %[[RET:.+]]  = arith.sitofp %[[EXTR]] : i16 to f16
// CHECK-NEXT:    return %[[RET]] : f16
func.func @extsi_over_extractelement_3xi16(%a: vector<3xi16>, %pos: i32) -> f16 {
  %b = arith.extsi %a : vector<3xi16> to vector<3xi32>
  %c = vector.extractelement %b[%pos : i32] : vector<3xi32>
  %f = arith.sitofp %c : i32 to f16
  return %f : f16
}

// CHECK-LABEL: func.func @extui_over_extractelement_3xi16
// CHECK-SAME:    (%[[ARG:.+]]: vector<3xi16>, %[[POS:.+]]: i32)
// CHECK-NEXT:    %[[EXTR:.+]] = vector.extractelement %[[ARG]][%[[POS]] : i32] : vector<3xi16>
// CHECK-NEXT:    %[[RET:.+]]  = arith.uitofp %[[EXTR]] : i16 to f16
// CHECK-NEXT:    return %[[RET]] : f16
func.func @extui_over_extractelement_3xi16(%a: vector<3xi16>, %pos: i32) -> f16 {
  %b = arith.extui %a : vector<3xi16> to vector<3xi32>
  %c = vector.extractelement %b[%pos : i32] : vector<3xi32>
  %f = arith.uitofp %c : i32 to f16
  return %f : f16
}

// CHECK-LABEL: func.func @extsi_over_extract_strided_slice_1d
// CHECK-SAME:    (%[[ARG:.+]]: vector<3xi16>)
// CHECK-NEXT:    %[[EXTR:.+]] = vector.extract_strided_slice %[[ARG]] {offsets = [1], sizes = [2], strides = [1]} : vector<3xi16> to vector<2xi16>
// CHECK-NEXT:    %[[RET:.+]]  = arith.extsi %[[EXTR]] : vector<2xi16> to vector<2xi32>
// CHECK-NEXT:    return %[[RET]] : vector<2xi32>
func.func @extsi_over_extract_strided_slice_1d(%a: vector<3xi16>) -> vector<2xi32> {
  %b = arith.extsi %a : vector<3xi16> to vector<3xi32>
  %c = vector.extract_strided_slice %b
   {offsets = [1], sizes = [2], strides = [1]} : vector<3xi32> to vector<2xi32>
  return %c : vector<2xi32>
}

// CHECK-LABEL: func.func @extui_over_extract_strided_slice_1d
// CHECK-SAME:    (%[[ARG:.+]]: vector<3xi16>)
// CHECK-NEXT:    %[[EXTR:.+]] = vector.extract_strided_slice %[[ARG]] {offsets = [1], sizes = [2], strides = [1]} : vector<3xi16> to vector<2xi16>
// CHECK-NEXT:    %[[RET:.+]]  = arith.extui %[[EXTR]] : vector<2xi16> to vector<2xi32>
// CHECK-NEXT:    return %[[RET]] : vector<2xi32>
func.func @extui_over_extract_strided_slice_1d(%a: vector<3xi16>) -> vector<2xi32> {
  %b = arith.extui %a : vector<3xi16> to vector<3xi32>
  %c = vector.extract_strided_slice %b
   {offsets = [1], sizes = [2], strides = [1]} : vector<3xi32> to vector<2xi32>
  return %c : vector<2xi32>
}

// CHECK-LABEL: func.func @extsi_over_extract_strided_slice_2d
// CHECK-SAME:    (%[[ARG:.+]]: vector<2x3xi16>)
// CHECK-NEXT:    %[[EXTR:.+]] = vector.extract_strided_slice %arg0 {offsets = [1, 1], sizes = [1, 2], strides = [1, 1]} : vector<2x3xi16> to vector<1x2xi16>
// CHECK-NEXT:    %[[RET:.+]]  = arith.extsi %[[EXTR]] : vector<1x2xi16> to vector<1x2xi32>
// CHECK-NEXT:    return %[[RET]] : vector<1x2xi32>
func.func @extsi_over_extract_strided_slice_2d(%a: vector<2x3xi16>) -> vector<1x2xi32> {
  %b = arith.extsi %a : vector<2x3xi16> to vector<2x3xi32>
  %c = vector.extract_strided_slice %b
   {offsets = [1, 1], sizes = [1, 2], strides = [1, 1]} : vector<2x3xi32> to vector<1x2xi32>
  return %c : vector<1x2xi32>
}

// CHECK-LABEL: func.func @extui_over_extract_strided_slice_2d
// CHECK-SAME:    (%[[ARG:.+]]: vector<2x3xi16>)
// CHECK-NEXT:    %[[EXTR:.+]] = vector.extract_strided_slice %arg0 {offsets = [1, 1], sizes = [1, 2], strides = [1, 1]} : vector<2x3xi16> to vector<1x2xi16>
// CHECK-NEXT:    %[[RET:.+]]  = arith.extui %[[EXTR]] : vector<1x2xi16> to vector<1x2xi32>
// CHECK-NEXT:    return %[[RET]] : vector<1x2xi32>
func.func @extui_over_extract_strided_slice_2d(%a: vector<2x3xi16>) -> vector<1x2xi32> {
  %b = arith.extui %a : vector<2x3xi16> to vector<2x3xi32>
  %c = vector.extract_strided_slice %b
   {offsets = [1, 1], sizes = [1, 2], strides = [1, 1]} : vector<2x3xi32> to vector<1x2xi32>
  return %c : vector<1x2xi32>
}

// CHECK-LABEL: func.func @extsi_over_insert_3xi16
// CHECK-SAME:    (%[[ARG0:.+]]: vector<3xi16>, %[[ARG1:.+]]: i16)
// CHECK-NEXT:    %[[INS:.+]] = vector.insert %[[ARG1]], %[[ARG0]] [1] : i16 into vector<3xi16>
// CHECK-NEXT:    %[[RET:.+]] = arith.extsi %[[INS]] : vector<3xi16> to vector<3xi32>
// CHECK-NEXT:    return %[[RET]] : vector<3xi32>
func.func @extsi_over_insert_3xi16(%a: vector<3xi16>, %b: i16) -> vector<3xi32> {
  %c = arith.extsi %a : vector<3xi16> to vector<3xi32>
  %d = arith.extsi %b : i16 to i32
  %e = vector.insert %d, %c [1] : i32 into vector<3xi32>
  return %e : vector<3xi32>
}

// CHECK-LABEL: func.func @extui_over_insert_3xi16
// CHECK-SAME:    (%[[ARG0:.+]]: vector<3xi16>, %[[ARG1:.+]]: i16)
// CHECK-NEXT:    %[[INS:.+]] = vector.insert %[[ARG1]], %[[ARG0]] [1] : i16 into vector<3xi16>
// CHECK-NEXT:    %[[RET:.+]] = arith.extui %[[INS]] : vector<3xi16> to vector<3xi32>
// CHECK-NEXT:    return %[[RET]] : vector<3xi32>
func.func @extui_over_insert_3xi16(%a: vector<3xi16>, %b: i16) -> vector<3xi32> {
  %c = arith.extui %a : vector<3xi16> to vector<3xi32>
  %d = arith.extui %b : i16 to i32
  %e = vector.insert %d, %c [1] : i32 into vector<3xi32>
  return %e : vector<3xi32>
}

// CHECK-LABEL: func.func @extsi_over_insert_3xi16_cst_0
// CHECK-SAME:    (%[[ARG:.+]]: i16)
// CHECK-NEXT:    %[[CST:.+]] = arith.constant dense<0> : vector<3xi16>
// CHECK-NEXT:    %[[INS:.+]] = vector.insert %[[ARG]], %[[CST]] [1] : i16 into vector<3xi16>
// CHECK-NEXT:    %[[RET:.+]] = arith.extsi %[[INS]] : vector<3xi16> to vector<3xi32>
// CHECK-NEXT:    return %[[RET]] : vector<3xi32>
func.func @extsi_over_insert_3xi16_cst_0(%a: i16) -> vector<3xi32> {
  %cst = arith.constant dense<0> : vector<3xi32>
  %d = arith.extsi %a : i16 to i32
  %e = vector.insert %d, %cst [1] : i32 into vector<3xi32>
  return %e : vector<3xi32>
}

// CHECK-LABEL: func.func @extsi_over_insert_3xi8_cst
// CHECK-SAME:    (%[[ARG:.+]]: i8)
// CHECK-NEXT:    %[[CST:.+]] = arith.constant dense<[-1, 127, -128]> : vector<3xi8>
// CHECK-NEXT:    %[[INS:.+]] = vector.insert %[[ARG]], %[[CST]] [1] : i8 into vector<3xi8>
// CHECK-NEXT:    %[[RET:.+]] = arith.extsi %[[INS]] : vector<3xi8> to vector<3xi32>
// CHECK-NEXT:    return %[[RET]] : vector<3xi32>
func.func @extsi_over_insert_3xi8_cst(%a: i8) -> vector<3xi32> {
  %cst = arith.constant dense<[-1, 127, -128]> : vector<3xi32>
  %d = arith.extsi %a : i8 to i32
  %e = vector.insert %d, %cst [1] : i32 into vector<3xi32>
  return %e : vector<3xi32>
}

// CHECK-LABEL: func.func @extui_over_insert_3xi8_cst
// CHECK-SAME:    (%[[ARG:.+]]: i8)
// CHECK-NEXT:    %[[CST:.+]] = arith.constant dense<[1, 127, -1]> : vector<3xi8>
// CHECK-NEXT:    %[[INS:.+]] = vector.insert %[[ARG]], %[[CST]] [1] : i8 into vector<3xi8>
// CHECK-NEXT:    %[[RET:.+]] = arith.extui %[[INS]] : vector<3xi8> to vector<3xi32>
// CHECK-NEXT:    return %[[RET]] : vector<3xi32>
func.func @extui_over_insert_3xi8_cst(%a: i8) -> vector<3xi32> {
  %cst = arith.constant dense<[1, 127, 255]> : vector<3xi32>
  %d = arith.extui %a : i8 to i32
  %e = vector.insert %d, %cst [1] : i32 into vector<3xi32>
  return %e : vector<3xi32>
}

// CHECK-LABEL: func.func @extsi_over_insert_3xi16_cst_i16
// CHECK-SAME:    (%[[ARG:.+]]: i8)
// CHECK-NEXT:    %[[CST:.+]]  = arith.constant dense<[-1, 128, 0]> : vector<3xi16>
// CHECK-NEXT:    %[[SRCE:.+]] = arith.extsi %[[ARG]] : i8 to i32
// CHECK-NEXT:    %[[SRCT:.+]] = arith.trunci %[[SRCE]] : i32 to i16
// CHECK-NEXT:    %[[INS:.+]]  = vector.insert %[[SRCT]], %[[CST]] [1] : i16 into vector<3xi16>
// CHECK-NEXT:    %[[RET:.+]]  = arith.extsi %[[INS]] : vector<3xi16> to vector<3xi32>
// CHECK-NEXT:    return %[[RET]] : vector<3xi32>
func.func @extsi_over_insert_3xi16_cst_i16(%a: i8) -> vector<3xi32> {
  %cst = arith.constant dense<[-1, 128, 0]> : vector<3xi32>
  %d = arith.extsi %a : i8 to i32
  %e = vector.insert %d, %cst [1] : i32 into vector<3xi32>
  return %e : vector<3xi32>
}

// CHECK-LABEL: func.func @extui_over_insert_3xi16_cst_i16
// CHECK-SAME:    (%[[ARG:.+]]: i8)
// CHECK-NEXT:    %[[CST:.+]]  = arith.constant dense<[1, 256, 0]> : vector<3xi16>
// CHECK-NEXT:    %[[SRCE:.+]] = arith.extui %[[ARG]] : i8 to i32
// CHECK-NEXT:    %[[SRCT:.+]] = arith.trunci %[[SRCE]] : i32 to i16
// CHECK-NEXT:    %[[INS:.+]]  = vector.insert %[[SRCT]], %[[CST]] [1] : i16 into vector<3xi16>
// CHECK-NEXT:    %[[RET:.+]]  = arith.extui %[[INS]] : vector<3xi16> to vector<3xi32>
// CHECK-NEXT:    return %[[RET]] : vector<3xi32>
func.func @extui_over_insert_3xi16_cst_i16(%a: i8) -> vector<3xi32> {
  %cst = arith.constant dense<[1, 256, 0]> : vector<3xi32>
  %d = arith.extui %a : i8 to i32
  %e = vector.insert %d, %cst [1] : i32 into vector<3xi32>
  return %e : vector<3xi32>
}

// CHECK-LABEL: func.func @extsi_over_insertelement_3xi16
// CHECK-SAME:    (%[[ARG0:.+]]: vector<3xi16>, %[[ARG1:.+]]: i16, %[[POS:.+]]: i32)
// CHECK-NEXT:    %[[INS:.+]] = vector.insertelement %[[ARG1]], %[[ARG0]][%[[POS]] : i32] : vector<3xi16>
// CHECK-NEXT:    %[[RET:.+]] = arith.extsi %[[INS]] : vector<3xi16> to vector<3xi32>
// CHECK-NEXT:    return %[[RET]] : vector<3xi32>
func.func @extsi_over_insertelement_3xi16(%a: vector<3xi16>, %b: i16, %pos: i32) -> vector<3xi32> {
  %c = arith.extsi %a : vector<3xi16> to vector<3xi32>
  %d = arith.extsi %b : i16 to i32
  %e = vector.insertelement %d, %c[%pos : i32] : vector<3xi32>
  return %e : vector<3xi32>
}

// CHECK-LABEL: func.func @extui_over_insertelement_3xi16
// CHECK-SAME:    (%[[ARG0:.+]]: vector<3xi16>, %[[ARG1:.+]]: i16, %[[POS:.+]]: i32)
// CHECK-NEXT:    %[[INS:.+]] = vector.insertelement %[[ARG1]], %[[ARG0]][%[[POS]] : i32] : vector<3xi16>
// CHECK-NEXT:    %[[RET:.+]] = arith.extui %[[INS]] : vector<3xi16> to vector<3xi32>
// CHECK-NEXT:    return %[[RET]] : vector<3xi32>
func.func @extui_over_insertelement_3xi16(%a: vector<3xi16>, %b: i16, %pos: i32) -> vector<3xi32> {
  %c = arith.extui %a : vector<3xi16> to vector<3xi32>
  %d = arith.extui %b : i16 to i32
  %e = vector.insertelement %d, %c[%pos : i32] : vector<3xi32>
  return %e : vector<3xi32>
}

// CHECK-LABEL: func.func @extsi_over_insertelement_3xi16_cst_i16
// CHECK-SAME:    (%[[ARG:.+]]: i8, %[[POS:.+]]: i32)
// CHECK-NEXT:    %[[CST:.+]]  = arith.constant dense<[-1, 128, 0]> : vector<3xi16>
// CHECK-NEXT:    %[[SRCE:.+]] = arith.extsi %[[ARG]] : i8 to i32
// CHECK-NEXT:    %[[SRCT:.+]] = arith.trunci %[[SRCE]] : i32 to i16
// CHECK-NEXT:    %[[INS:.+]] = vector.insertelement %[[SRCT]], %[[CST]][%[[POS]] : i32] : vector<3xi16>
// CHECK-NEXT:    %[[RET:.+]]  = arith.extsi %[[INS]] : vector<3xi16> to vector<3xi32>
// CHECK-NEXT:    return %[[RET]] : vector<3xi32>
func.func @extsi_over_insertelement_3xi16_cst_i16(%a: i8, %pos: i32) -> vector<3xi32> {
  %cst = arith.constant dense<[-1, 128, 0]> : vector<3xi32>
  %d = arith.extsi %a : i8 to i32
  %e = vector.insertelement %d, %cst[%pos : i32] : vector<3xi32>
  return %e : vector<3xi32>
}

// CHECK-LABEL: func.func @extui_over_insertelement_3xi16_cst_i16
// CHECK-SAME:    (%[[ARG:.+]]: i8, %[[POS:.+]]: i32)
// CHECK-NEXT:    %[[CST:.+]]  = arith.constant dense<[1, 256, 0]> : vector<3xi16>
// CHECK-NEXT:    %[[SRCE:.+]] = arith.extui %[[ARG]] : i8 to i32
// CHECK-NEXT:    %[[SRCT:.+]] = arith.trunci %[[SRCE]] : i32 to i16
// CHECK-NEXT:    %[[INS:.+]] = vector.insertelement %[[SRCT]], %[[CST]][%[[POS]] : i32] : vector<3xi16>
// CHECK-NEXT:    %[[RET:.+]]  = arith.extui %[[INS]] : vector<3xi16> to vector<3xi32>
// CHECK-NEXT:    return %[[RET]] : vector<3xi32>
func.func @extui_over_insertelement_3xi16_cst_i16(%a: i8, %pos: i32) -> vector<3xi32> {
  %cst = arith.constant dense<[1, 256, 0]> : vector<3xi32>
  %d = arith.extui %a : i8 to i32
  %e = vector.insertelement %d, %cst[%pos : i32] : vector<3xi32>
  return %e : vector<3xi32>
}

// CHECK-LABEL: func.func @extsi_over_insert_strided_slice_1d
// CHECK-SAME:    (%[[ARG0:.+]]: vector<3xi16>, %[[ARG1:.+]]: vector<2xi16>)
// CHECK-NEXT:    %[[INS:.+]] = vector.insert_strided_slice %[[ARG1]], %[[ARG0]]
// CHECK-SAME:                    {offsets = [1], strides = [1]} : vector<2xi16> into vector<3xi16>
// CHECK-NEXT:    %[[RET:.+]] = arith.extsi %[[INS]] : vector<3xi16> to vector<3xi32>
// CHECK-NEXT:    return %[[RET]] : vector<3xi32>
func.func @extsi_over_insert_strided_slice_1d(%a: vector<3xi16>, %b: vector<2xi16>) -> vector<3xi32> {
  %c = arith.extsi %a : vector<3xi16> to vector<3xi32>
  %d = arith.extsi %b : vector<2xi16> to vector<2xi32>
  %e = vector.insert_strided_slice %d, %c {offsets = [1], strides = [1]} : vector<2xi32> into vector<3xi32>
  return %e : vector<3xi32>
}

// CHECK-LABEL: func.func @extui_over_insert_strided_slice_1d
// CHECK-SAME:    (%[[ARG0:.+]]: vector<3xi16>, %[[ARG1:.+]]: vector<2xi16>)
// CHECK-NEXT:    %[[INS:.+]] = vector.insert_strided_slice %[[ARG1]], %[[ARG0]]
// CHECK-SAME:                    {offsets = [1], strides = [1]} : vector<2xi16> into vector<3xi16>
// CHECK-NEXT:    %[[RET:.+]] = arith.extui %[[INS]] : vector<3xi16> to vector<3xi32>
// CHECK-NEXT:    return %[[RET]] : vector<3xi32>
func.func @extui_over_insert_strided_slice_1d(%a: vector<3xi16>, %b: vector<2xi16>) -> vector<3xi32> {
  %c = arith.extui %a : vector<3xi16> to vector<3xi32>
  %d = arith.extui %b : vector<2xi16> to vector<2xi32>
  %e = vector.insert_strided_slice %d, %c {offsets = [1], strides = [1]} : vector<2xi32> into vector<3xi32>
  return %e : vector<3xi32>
}

// CHECK-LABEL: func.func @extsi_over_insert_strided_slice_cst_2d
// CHECK-SAME:    (%[[ARG:.+]]: vector<1x2xi8>)
// CHECK-NEXT:    %[[CST:.+]]  = arith.constant
// CHECK-SAME{LITERAL}:            dense<[[-1, 128, 0], [-129, 42, 1337]]> : vector<2x3xi16>
// CHECK-NEXT:    %[[SRCE:.+]] = arith.extsi %[[ARG]] : vector<1x2xi8> to vector<1x2xi32>
// CHECK-NEXT:    %[[SRCT:.+]] = arith.trunci %[[SRCE]] : vector<1x2xi32> to vector<1x2xi16>
// CHECK-NEXT:    %[[INS:.+]] = vector.insert_strided_slice %[[SRCT]], %[[CST]]
// CHECK-SAME:                    {offsets = [0, 1], strides = [1, 1]} : vector<1x2xi16> into vector<2x3xi16>
// CHECK-NEXT:    %[[RET:.+]]  = arith.extsi %[[INS]] : vector<2x3xi16> to vector<2x3xi32>
// CHECK-NEXT:    return %[[RET]] : vector<2x3xi32>
func.func @extsi_over_insert_strided_slice_cst_2d(%a: vector<1x2xi8>) -> vector<2x3xi32> {
  %cst = arith.constant dense<[[-1, 128, 0], [-129, 42, 1337]]> : vector<2x3xi32>
  %d = arith.extsi %a : vector<1x2xi8> to vector<1x2xi32>
  %e = vector.insert_strided_slice %d, %cst {offsets = [0, 1], strides = [1, 1]} : vector<1x2xi32> into vector<2x3xi32>
  return %e : vector<2x3xi32>
}

// CHECK-LABEL: func.func @extui_over_insert_strided_slice_cst_2d
// CHECK-SAME:    (%[[ARG:.+]]: vector<1x2xi8>)
// CHECK-NEXT:    %[[CST:.+]]  = arith.constant
// CHECK-SAME{LITERAL}:            dense<[[1, 128, 0], [256, 42, 1337]]> : vector<2x3xi16>
// CHECK-NEXT:    %[[SRCE:.+]] = arith.extui %[[ARG]] : vector<1x2xi8> to vector<1x2xi32>
// CHECK-NEXT:    %[[SRCT:.+]] = arith.trunci %[[SRCE]] : vector<1x2xi32> to vector<1x2xi16>
// CHECK-NEXT:    %[[INS:.+]] = vector.insert_strided_slice %[[SRCT]], %[[CST]]
// CHECK-SAME:                    {offsets = [0, 1], strides = [1, 1]} : vector<1x2xi16> into vector<2x3xi16>
// CHECK-NEXT:    %[[RET:.+]]  = arith.extui %[[INS]] : vector<2x3xi16> to vector<2x3xi32>
// CHECK-NEXT:    return %[[RET]] : vector<2x3xi32>
func.func @extui_over_insert_strided_slice_cst_2d(%a: vector<1x2xi8>) -> vector<2x3xi32> {
  %cst = arith.constant dense<[[1, 128, 0], [256, 42, 1337]]> : vector<2x3xi32>
  %d = arith.extui %a : vector<1x2xi8> to vector<1x2xi32>
  %e = vector.insert_strided_slice %d, %cst {offsets = [0, 1], strides = [1, 1]} : vector<1x2xi32> into vector<2x3xi32>
  return %e : vector<2x3xi32>
}

// CHECK-LABEL: func.func @extsi_over_broadcast_3xi16
// CHECK-SAME:    (%[[ARG:.+]]: i16)
// CHECK-NEXT:    %[[BCST:.+]] = vector.broadcast %[[ARG]] : i16 to vector<3xi16>
// CHECK-NEXT:    %[[RET:.+]]  = arith.extsi %[[BCST]] : vector<3xi16> to vector<3xi32>
// CHECK-NEXT:    return %[[RET]] : vector<3xi32>
func.func @extsi_over_broadcast_3xi16(%a: i16) -> vector<3xi32> {
  %b = arith.extsi %a : i16 to i32
  %r = vector.broadcast %b : i32 to vector<3xi32>
  return %r : vector<3xi32>
}

// CHECK-LABEL: func.func @extui_over_broadcast_2x3xi16
// CHECK-SAME:    (%[[ARG:.+]]: vector<3xi16>)
// CHECK-NEXT:    %[[BCST:.+]] = vector.broadcast %[[ARG]] : vector<3xi16> to vector<2x3xi16>
// CHECK-NEXT:    %[[RET:.+]]  = arith.extui %[[BCST]] : vector<2x3xi16> to vector<2x3xi32>
// CHECK-NEXT:    return %[[RET]] : vector<2x3xi32>
func.func @extui_over_broadcast_2x3xi16(%a: vector<3xi16>) -> vector<2x3xi32> {
  %b = arith.extui %a : vector<3xi16> to vector<3xi32>
  %r = vector.broadcast %b : vector<3xi32> to vector<2x3xi32>
  return %r : vector<2x3xi32>
}

// CHECK-LABEL: func.func @extsi_over_shape_cast_2x3xi16
// CHECK-SAME:    (%[[ARG:.+]]: vector<2x3xi16>)
// CHECK-NEXT:    %[[CAST:.+]] = vector.shape_cast %[[ARG]] : vector<2x3xi16> to vector<3x2xi16>
// CHECK-NEXT:    %[[RET:.+]]  = arith.extsi %[[CAST]] : vector<3x2xi16> to vector<3x2xi32>
// CHECK-NEXT:    return %[[RET]] : vector<3x2xi32>
func.func @extsi_over_shape_cast_2x3xi16(%a: vector<2x3xi16>) -> vector<3x2xi32> {
  %b = arith.extsi %a : vector<2x3xi16> to vector<2x3xi32>
  %r = vector.shape_cast %b : vector<2x3xi32> to vector<3x2xi32>
  return %r : vector<3x2xi32>
}

// CHECK-LABEL: func.func @extui_over_shape_cast_5x2x3xi16
// CHECK-SAME:    (%[[ARG:.+]]: vector<5x2x3xi16>)
// CHECK-NEXT:    %[[CAST:.+]] = vector.shape_cast %[[ARG]] : vector<5x2x3xi16> to vector<2x3x5xi16>
// CHECK-NEXT:    %[[RET:.+]]  = arith.extui %[[CAST]] : vector<2x3x5xi16> to vector<2x3x5xi32>
// CHECK-NEXT:    return %[[RET]] : vector<2x3x5xi32>
func.func @extui_over_shape_cast_5x2x3xi16(%a: vector<5x2x3xi16>) -> vector<2x3x5xi32> {
  %b = arith.extui %a : vector<5x2x3xi16> to vector<5x2x3xi32>
  %r = vector.shape_cast %b : vector<5x2x3xi32> to vector<2x3x5xi32>
  return %r : vector<2x3x5xi32>
}

// CHECK-LABEL: func.func @extsi_over_transpose_2x3xi16
// CHECK-SAME:    (%[[ARG:.+]]: vector<2x3xi16>)
// CHECK-NEXT:    %[[TRAN:.+]] = vector.transpose %[[ARG]], [1, 0] : vector<2x3xi16> to vector<3x2xi16>
// CHECK-NEXT:    %[[RET:.+]]  = arith.extsi %[[TRAN]] : vector<3x2xi16> to vector<3x2xi32>
// CHECK-NEXT:    return %[[RET]] : vector<3x2xi32>
func.func @extsi_over_transpose_2x3xi16(%a: vector<2x3xi16>) -> vector<3x2xi32> {
  %b = arith.extsi %a : vector<2x3xi16> to vector<2x3xi32>
  %r = vector.transpose %b, [1, 0] : vector<2x3xi32> to vector<3x2xi32>
  return %r : vector<3x2xi32>
}

// CHECK-LABEL: func.func @extui_over_transpose_5x2x3xi16
// CHECK-SAME:    (%[[ARG:.+]]: vector<5x2x3xi16>)
// CHECK-NEXT:    %[[TRAN:.+]] = vector.transpose %[[ARG]], [1, 2, 0] : vector<5x2x3xi16> to vector<2x3x5xi16>
// CHECK-NEXT:    %[[RET:.+]]  = arith.extui %[[TRAN]] : vector<2x3x5xi16> to vector<2x3x5xi32>
// CHECK-NEXT:    return %[[RET]] : vector<2x3x5xi32>
func.func @extui_over_transpose_5x2x3xi16(%a: vector<5x2x3xi16>) -> vector<2x3x5xi32> {
  %b = arith.extui %a : vector<5x2x3xi16> to vector<5x2x3xi32>
  %r = vector.transpose %b, [1, 2, 0] : vector<5x2x3xi32> to vector<2x3x5xi32>
  return %r : vector<2x3x5xi32>
}

// CHECK-LABEL: func.func @extsi_over_flat_transpose_16xi16
// CHECK-SAME:    (%[[ARG:.+]]: vector<16xi16>)
// CHECK-NEXT:    %[[TRAN:.+]] = vector.flat_transpose %[[ARG]] {columns = 4 : i32, rows = 4 : i32} : vector<16xi16> -> vector<16xi16>
// CHECK-NEXT:    %[[RET:.+]]  = arith.extsi %[[TRAN]] : vector<16xi16> to vector<16xi32>
// CHECK-NEXT:    return %[[RET]] : vector<16xi32>
func.func @extsi_over_flat_transpose_16xi16(%a: vector<16xi16>) -> vector<16xi32> {
  %b = arith.extsi %a : vector<16xi16> to vector<16xi32>
  %r = vector.flat_transpose %b {columns = 4 : i32, rows = 4 : i32} : vector<16xi32> -> vector<16xi32>
  return %r : vector<16xi32>
}

// CHECK-LABEL: func.func @extui_over_flat_transpose_16xi16
// CHECK-SAME:    (%[[ARG:.+]]: vector<16xi16>)
// CHECK-NEXT:    %[[TRAN:.+]] = vector.flat_transpose %[[ARG]] {columns = 8 : i32, rows = 2 : i32} : vector<16xi16> -> vector<16xi16>
// CHECK-NEXT:    %[[RET:.+]]  = arith.extui %[[TRAN]] : vector<16xi16> to vector<16xi32>
// CHECK-NEXT:    return %[[RET]] : vector<16xi32>
func.func @extui_over_flat_transpose_16xi16(%a: vector<16xi16>) -> vector<16xi32> {
  %b = arith.extui %a : vector<16xi16> to vector<16xi32>
  %r = vector.flat_transpose %b {columns = 8 : i32, rows = 2 : i32} : vector<16xi32> -> vector<16xi32>
  return %r : vector<16xi32>
}
