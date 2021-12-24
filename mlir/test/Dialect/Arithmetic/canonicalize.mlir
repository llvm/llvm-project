// RUN: mlir-opt %s -canonicalize --split-input-file | FileCheck %s

// Test case: Folding of comparisons with equal operands.
// CHECK-LABEL: @cmpi_equal_operands
//   CHECK-DAG:   %[[T:.*]] = arith.constant true
//   CHECK-DAG:   %[[F:.*]] = arith.constant false
//       CHECK:   return %[[T]], %[[T]], %[[T]], %[[T]], %[[T]],
//  CHECK-SAME:          %[[F]], %[[F]], %[[F]], %[[F]], %[[F]]
func @cmpi_equal_operands(%arg0: i64)
    -> (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1) {
  %0 = arith.cmpi eq, %arg0, %arg0 : i64
  %1 = arith.cmpi sle, %arg0, %arg0 : i64
  %2 = arith.cmpi sge, %arg0, %arg0 : i64
  %3 = arith.cmpi ule, %arg0, %arg0 : i64
  %4 = arith.cmpi uge, %arg0, %arg0 : i64
  %5 = arith.cmpi ne, %arg0, %arg0 : i64
  %6 = arith.cmpi slt, %arg0, %arg0 : i64
  %7 = arith.cmpi sgt, %arg0, %arg0 : i64
  %8 = arith.cmpi ult, %arg0, %arg0 : i64
  %9 = arith.cmpi ugt, %arg0, %arg0 : i64
  return %0, %1, %2, %3, %4, %5, %6, %7, %8, %9
      : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
}

// Test case: Folding of comparisons with equal vector operands.
// CHECK-LABEL: @cmpi_equal_vector_operands
//   CHECK-DAG:   %[[T:.*]] = arith.constant dense<true>
//   CHECK-DAG:   %[[F:.*]] = arith.constant dense<false>
//       CHECK:   return %[[T]], %[[T]], %[[T]], %[[T]], %[[T]],
//  CHECK-SAME:          %[[F]], %[[F]], %[[F]], %[[F]], %[[F]]
func @cmpi_equal_vector_operands(%arg0: vector<1x8xi64>)
    -> (vector<1x8xi1>, vector<1x8xi1>, vector<1x8xi1>, vector<1x8xi1>,
        vector<1x8xi1>, vector<1x8xi1>, vector<1x8xi1>, vector<1x8xi1>,
	vector<1x8xi1>, vector<1x8xi1>) {
  %0 = arith.cmpi eq, %arg0, %arg0 : vector<1x8xi64>
  %1 = arith.cmpi sle, %arg0, %arg0 : vector<1x8xi64>
  %2 = arith.cmpi sge, %arg0, %arg0 : vector<1x8xi64>
  %3 = arith.cmpi ule, %arg0, %arg0 : vector<1x8xi64>
  %4 = arith.cmpi uge, %arg0, %arg0 : vector<1x8xi64>
  %5 = arith.cmpi ne, %arg0, %arg0 : vector<1x8xi64>
  %6 = arith.cmpi slt, %arg0, %arg0 : vector<1x8xi64>
  %7 = arith.cmpi sgt, %arg0, %arg0 : vector<1x8xi64>
  %8 = arith.cmpi ult, %arg0, %arg0 : vector<1x8xi64>
  %9 = arith.cmpi ugt, %arg0, %arg0 : vector<1x8xi64>
  return %0, %1, %2, %3, %4, %5, %6, %7, %8, %9
      : vector<1x8xi1>, vector<1x8xi1>, vector<1x8xi1>, vector<1x8xi1>,
        vector<1x8xi1>, vector<1x8xi1>, vector<1x8xi1>, vector<1x8xi1>,
	vector<1x8xi1>, vector<1x8xi1>
}

// -----

// CHECK-LABEL: @indexCastOfSignExtend
//       CHECK:   %[[res:.+]] = arith.index_cast %arg0 : i8 to index
//       CHECK:   return %[[res]]
func @indexCastOfSignExtend(%arg0: i8) -> index {
  %ext = arith.extsi %arg0 : i8 to i16
  %idx = arith.index_cast %ext : i16 to index
  return %idx : index
}

// CHECK-LABEL: @signExtendConstant
//       CHECK:   %[[cres:.+]] = arith.constant -2 : i16
//       CHECK:   return %[[cres]]
func @signExtendConstant() -> i16 {
  %c-2 = arith.constant -2 : i8
  %ext = arith.extsi %c-2 : i8 to i16
  return %ext : i16
}

// CHECK-LABEL: @truncConstant
//       CHECK:   %[[cres:.+]] = arith.constant -2 : i16
//       CHECK:   return %[[cres]]
func @truncConstant(%arg0: i8) -> i16 {
  %c-2 = arith.constant -2 : i32
  %tr = arith.trunci %c-2 : i32 to i16
  return %tr : i16
}

// CHECK-LABEL: @truncFPConstant
//       CHECK:   %[[cres:.+]] = arith.constant 1.000000e+00 : bf16
//       CHECK:   return %[[cres]]
func @truncFPConstant() -> bf16 {
  %cst = arith.constant 1.000000e+00 : f32
  %0 = arith.truncf %cst : f32 to bf16
  return %0 : bf16
}

// Test that cases with rounding are NOT propagated
// CHECK-LABEL: @truncFPConstantRounding
//       CHECK:   arith.constant 1.444000e+25 : f32
//       CHECK:   truncf
func @truncFPConstantRounding() -> bf16 {
  %cst = arith.constant 1.444000e+25 : f32
  %0 = arith.truncf %cst : f32 to bf16
  return %0 : bf16
}

// CHECK-LABEL: @tripleAddAdd
//       CHECK:   %[[cres:.+]] = arith.constant 59 : index
//       CHECK:   %[[add:.+]] = arith.addi %arg0, %[[cres]] : index
//       CHECK:   return %[[add]]
func @tripleAddAdd(%arg0: index) -> index {
  %c17 = arith.constant 17 : index
  %c42 = arith.constant 42 : index
  %add1 = arith.addi %c17, %arg0 : index
  %add2 = arith.addi %c42, %add1 : index
  return %add2 : index
}

// CHECK-LABEL: @tripleAddSub0
//       CHECK:   %[[cres:.+]] = arith.constant 59 : index
//       CHECK:   %[[add:.+]] = arith.subi %[[cres]], %arg0 : index
//       CHECK:   return %[[add]]
func @tripleAddSub0(%arg0: index) -> index {
  %c17 = arith.constant 17 : index
  %c42 = arith.constant 42 : index
  %add1 = arith.subi %c17, %arg0 : index
  %add2 = arith.addi %c42, %add1 : index
  return %add2 : index
}

// CHECK-LABEL: @tripleAddSub1
//       CHECK:   %[[cres:.+]] = arith.constant 25 : index
//       CHECK:   %[[add:.+]] = arith.addi %arg0, %[[cres]] : index
//       CHECK:   return %[[add]]
func @tripleAddSub1(%arg0: index) -> index {
  %c17 = arith.constant 17 : index
  %c42 = arith.constant 42 : index
  %add1 = arith.subi %arg0, %c17 : index
  %add2 = arith.addi %c42, %add1 : index
  return %add2 : index
}

// CHECK-LABEL: @tripleSubAdd0
//       CHECK:   %[[cres:.+]] = arith.constant 25 : index
//       CHECK:   %[[add:.+]] = arith.subi %[[cres]], %arg0 : index
//       CHECK:   return %[[add]]
func @tripleSubAdd0(%arg0: index) -> index {
  %c17 = arith.constant 17 : index
  %c42 = arith.constant 42 : index
  %add1 = arith.addi %c17, %arg0 : index
  %add2 = arith.subi %c42, %add1 : index
  return %add2 : index
}

// CHECK-LABEL: @tripleSubAdd1
//       CHECK:   %[[cres:.+]] = arith.constant -25 : index
//       CHECK:   %[[add:.+]] = arith.addi %arg0, %[[cres]] : index
//       CHECK:   return %[[add]]
func @tripleSubAdd1(%arg0: index) -> index {
  %c17 = arith.constant 17 : index
  %c42 = arith.constant 42 : index
  %add1 = arith.addi %c17, %arg0 : index
  %add2 = arith.subi %add1, %c42 : index
  return %add2 : index
}

// CHECK-LABEL: @tripleSubSub0
//       CHECK:   %[[cres:.+]] = arith.constant 25 : index
//       CHECK:   %[[add:.+]] = arith.addi %arg0, %[[cres]] : index
//       CHECK:   return %[[add]]
func @tripleSubSub0(%arg0: index) -> index {
  %c17 = arith.constant 17 : index
  %c42 = arith.constant 42 : index
  %add1 = arith.subi %c17, %arg0 : index
  %add2 = arith.subi %c42, %add1 : index
  return %add2 : index
}

// CHECK-LABEL: @tripleSubSub1
//       CHECK:   %[[cres:.+]] = arith.constant -25 : index
//       CHECK:   %[[add:.+]] = arith.subi %[[cres]], %arg0 : index
//       CHECK:   return %[[add]]
func @tripleSubSub1(%arg0: index) -> index {
  %c17 = arith.constant 17 : index
  %c42 = arith.constant 42 : index
  %add1 = arith.subi %c17, %arg0 : index
  %add2 = arith.subi %add1, %c42 : index
  return %add2 : index
}

// CHECK-LABEL: @tripleSubSub2
//       CHECK:   %[[cres:.+]] = arith.constant 59 : index
//       CHECK:   %[[add:.+]] = arith.subi %[[cres]], %arg0 : index
//       CHECK:   return %[[add]]
func @tripleSubSub2(%arg0: index) -> index {
  %c17 = arith.constant 17 : index
  %c42 = arith.constant 42 : index
  %add1 = arith.subi %arg0, %c17 : index
  %add2 = arith.subi %c42, %add1 : index
  return %add2 : index
}

// CHECK-LABEL: @tripleSubSub3
//       CHECK:   %[[cres:.+]] = arith.constant 59 : index
//       CHECK:   %[[add:.+]] = arith.subi %arg0, %[[cres]] : index
//       CHECK:   return %[[add]]
func @tripleSubSub3(%arg0: index) -> index {
  %c17 = arith.constant 17 : index
  %c42 = arith.constant 42 : index
  %add1 = arith.subi %arg0, %c17 : index
  %add2 = arith.subi %add1, %c42 : index
  return %add2 : index
}

// CHECK-LABEL: @notCmpEQ
//       CHECK:   %[[cres:.+]] = arith.cmpi ne, %arg0, %arg1 : i8
//       CHECK:   return %[[cres]]
func @notCmpEQ(%arg0: i8, %arg1: i8) -> i1 {
  %true = arith.constant true
  %cmp = arith.cmpi "eq", %arg0, %arg1 : i8
  %ncmp = arith.xori %cmp, %true : i1
  return %ncmp : i1
}

// CHECK-LABEL: @notCmpEQ2
//       CHECK:   %[[cres:.+]] = arith.cmpi ne, %arg0, %arg1 : i8
//       CHECK:   return %[[cres]]
func @notCmpEQ2(%arg0: i8, %arg1: i8) -> i1 {
  %true = arith.constant true
  %cmp = arith.cmpi "eq", %arg0, %arg1 : i8
  %ncmp = arith.xori %true, %cmp : i1
  return %ncmp : i1
}

// CHECK-LABEL: @notCmpNE
//       CHECK:   %[[cres:.+]] = arith.cmpi eq, %arg0, %arg1 : i8
//       CHECK:   return %[[cres]]
func @notCmpNE(%arg0: i8, %arg1: i8) -> i1 {
  %true = arith.constant true
  %cmp = arith.cmpi "ne", %arg0, %arg1 : i8
  %ncmp = arith.xori %cmp, %true : i1
  return %ncmp : i1
}

// CHECK-LABEL: @notCmpSLT
//       CHECK:   %[[cres:.+]] = arith.cmpi sge, %arg0, %arg1 : i8
//       CHECK:   return %[[cres]]
func @notCmpSLT(%arg0: i8, %arg1: i8) -> i1 {
  %true = arith.constant true
  %cmp = arith.cmpi "slt", %arg0, %arg1 : i8
  %ncmp = arith.xori %cmp, %true : i1
  return %ncmp : i1
}

// CHECK-LABEL: @notCmpSLE
//       CHECK:   %[[cres:.+]] = arith.cmpi sgt, %arg0, %arg1 : i8
//       CHECK:   return %[[cres]]
func @notCmpSLE(%arg0: i8, %arg1: i8) -> i1 {
  %true = arith.constant true
  %cmp = arith.cmpi "sle", %arg0, %arg1 : i8
  %ncmp = arith.xori %cmp, %true : i1
  return %ncmp : i1
}

// CHECK-LABEL: @notCmpSGT
//       CHECK:   %[[cres:.+]] = arith.cmpi sle, %arg0, %arg1 : i8
//       CHECK:   return %[[cres]]
func @notCmpSGT(%arg0: i8, %arg1: i8) -> i1 {
  %true = arith.constant true
  %cmp = arith.cmpi "sgt", %arg0, %arg1 : i8
  %ncmp = arith.xori %cmp, %true : i1
  return %ncmp : i1
}

// CHECK-LABEL: @notCmpSGE
//       CHECK:   %[[cres:.+]] = arith.cmpi slt, %arg0, %arg1 : i8
//       CHECK:   return %[[cres]]
func @notCmpSGE(%arg0: i8, %arg1: i8) -> i1 {
  %true = arith.constant true
  %cmp = arith.cmpi "sge", %arg0, %arg1 : i8
  %ncmp = arith.xori %cmp, %true : i1
  return %ncmp : i1
}

// CHECK-LABEL: @notCmpULT
//       CHECK:   %[[cres:.+]] = arith.cmpi uge, %arg0, %arg1 : i8
//       CHECK:   return %[[cres]]
func @notCmpULT(%arg0: i8, %arg1: i8) -> i1 {
  %true = arith.constant true
  %cmp = arith.cmpi "ult", %arg0, %arg1 : i8
  %ncmp = arith.xori %cmp, %true : i1
  return %ncmp : i1
}

// CHECK-LABEL: @notCmpULE
//       CHECK:   %[[cres:.+]] = arith.cmpi ugt, %arg0, %arg1 : i8
//       CHECK:   return %[[cres]]
func @notCmpULE(%arg0: i8, %arg1: i8) -> i1 {
  %true = arith.constant true
  %cmp = arith.cmpi "ule", %arg0, %arg1 : i8
  %ncmp = arith.xori %cmp, %true : i1
  return %ncmp : i1
}

// CHECK-LABEL: @notCmpUGT
//       CHECK:   %[[cres:.+]] = arith.cmpi ule, %arg0, %arg1 : i8
//       CHECK:   return %[[cres]]
func @notCmpUGT(%arg0: i8, %arg1: i8) -> i1 {
  %true = arith.constant true
  %cmp = arith.cmpi "ugt", %arg0, %arg1 : i8
  %ncmp = arith.xori %cmp, %true : i1
  return %ncmp : i1
}

// CHECK-LABEL: @notCmpUGE
//       CHECK:   %[[cres:.+]] = arith.cmpi ult, %arg0, %arg1 : i8
//       CHECK:   return %[[cres]]
func @notCmpUGE(%arg0: i8, %arg1: i8) -> i1 {
  %true = arith.constant true
  %cmp = arith.cmpi "uge", %arg0, %arg1 : i8
  %ncmp = arith.xori %cmp, %true : i1
  return %ncmp : i1
}

// -----

// CHECK-LABEL: @bitcastSameType(
// CHECK-SAME: %[[ARG:[a-zA-Z0-9_]*]]
func @bitcastSameType(%arg : f32) -> f32 {
  // CHECK: return %[[ARG]]
  %res = arith.bitcast %arg : f32 to f32
  return %res : f32
}

// -----

// CHECK-LABEL: @bitcastConstantFPtoI(
func @bitcastConstantFPtoI() -> i32 {
  // CHECK: %[[C0:.+]] = arith.constant 0 : i32
  // CHECK: return %[[C0]]
  %c0 = arith.constant 0.0 : f32
  %res = arith.bitcast %c0 : f32 to i32
  return %res : i32
}

// -----

// CHECK-LABEL: @bitcastConstantItoFP(
func @bitcastConstantItoFP() -> f32 {
  // CHECK: %[[C0:.+]] = arith.constant 0.0{{.*}} : f32
  // CHECK: return %[[C0]]
  %c0 = arith.constant 0 : i32
  %res = arith.bitcast %c0 : i32 to f32
  return %res : f32
}

// -----

// CHECK-LABEL: @bitcastConstantFPtoFP(
func @bitcastConstantFPtoFP() -> f16 {
  // CHECK: %[[C0:.+]] = arith.constant 0.0{{.*}} : f16
  // CHECK: return %[[C0]]
  %c0 = arith.constant 0.0 : bf16
  %res = arith.bitcast %c0 : bf16 to f16
  return %res : f16
}

// -----

// CHECK-LABEL: @bitcastConstantVecFPtoI(
func @bitcastConstantVecFPtoI() -> vector<3xf32> {
  // CHECK: %[[C0:.+]] = arith.constant dense<0.0{{.*}}> : vector<3xf32>
  // CHECK: return %[[C0]]
  %c0 = arith.constant dense<0> : vector<3xi32>
  %res = arith.bitcast %c0 : vector<3xi32> to vector<3xf32>
  return %res : vector<3xf32>
}

// -----

// CHECK-LABEL: @bitcastConstantVecItoFP(
func @bitcastConstantVecItoFP() -> vector<3xi32> {
  // CHECK: %[[C0:.+]] = arith.constant dense<0> : vector<3xi32>
  // CHECK: return %[[C0]]
  %c0 = arith.constant dense<0.0> : vector<3xf32>
  %res = arith.bitcast %c0 : vector<3xf32> to vector<3xi32>
  return %res : vector<3xi32>
}

// -----

// CHECK-LABEL: @bitcastConstantVecFPtoFP(
func @bitcastConstantVecFPtoFP() -> vector<3xbf16> {
  // CHECK: %[[C0:.+]] = arith.constant dense<0.0{{.*}}> : vector<3xbf16>
  // CHECK: return %[[C0]]
  %c0 = arith.constant dense<0.0> : vector<3xf16>
  %res = arith.bitcast %c0 : vector<3xf16> to vector<3xbf16>
  return %res : vector<3xbf16>
}

// -----

// CHECK-LABEL: @bitcastBackAndForth(
// CHECK-SAME: %[[ARG:[a-zA-Z0-9_]*]]
func @bitcastBackAndForth(%arg : i32) -> i32 {
  // CHECK: return %[[ARG]]
  %f = arith.bitcast %arg : i32 to f32
  %res = arith.bitcast %f : f32 to i32
  return %res : i32
}

// -----

// CHECK-LABEL: @bitcastOfBitcast(
// CHECK-SAME: %[[ARG:[a-zA-Z0-9_]*]]
func @bitcastOfBitcast(%arg : i16) -> i16 {
  // CHECK: return %[[ARG]]
  %f = arith.bitcast %arg : i16 to f16
  %bf = arith.bitcast %f : f16 to bf16
  %res = arith.bitcast %bf : bf16 to i16
  return %res : i16
}

// -----

// CHECK-LABEL: test_maxsi
// CHECK: %[[C0:.+]] = arith.constant 42
// CHECK: %[[MAX_INT_CST:.+]] = arith.constant 127
// CHECK: %[[X:.+]] = arith.maxsi %arg0, %[[C0]]
// CHECK: return %arg0, %[[MAX_INT_CST]], %arg0, %[[X]]
func @test_maxsi(%arg0 : i8) -> (i8, i8, i8, i8) {
  %maxIntCst = arith.constant 127 : i8
  %minIntCst = arith.constant -128 : i8
  %c0 = arith.constant 42 : i8
  %0 = arith.maxsi %arg0, %arg0 : i8
  %1 = arith.maxsi %arg0, %maxIntCst : i8
  %2 = arith.maxsi %arg0, %minIntCst : i8
  %3 = arith.maxsi %arg0, %c0 : i8
  return %0, %1, %2, %3: i8, i8, i8, i8
}

// -----

// CHECK-LABEL: test_maxui
// CHECK: %[[C0:.+]] = arith.constant 42
// CHECK: %[[MAX_INT_CST:.+]] = arith.constant -1
// CHECK: %[[X:.+]] = arith.maxui %arg0, %[[C0]]
// CHECK: return %arg0, %[[MAX_INT_CST]], %arg0, %[[X]]
func @test_maxui(%arg0 : i8) -> (i8, i8, i8, i8) {
  %maxIntCst = arith.constant 255 : i8
  %minIntCst = arith.constant 0 : i8
  %c0 = arith.constant 42 : i8
  %0 = arith.maxui %arg0, %arg0 : i8
  %1 = arith.maxui %arg0, %maxIntCst : i8
  %2 = arith.maxui %arg0, %minIntCst : i8
  %3 = arith.maxui %arg0, %c0 : i8
  return %0, %1, %2, %3: i8, i8, i8, i8
}

// -----

// CHECK-LABEL: test_minsi
// CHECK: %[[C0:.+]] = arith.constant 42
// CHECK: %[[MIN_INT_CST:.+]] = arith.constant -128
// CHECK: %[[X:.+]] = arith.minsi %arg0, %[[C0]]
// CHECK: return %arg0, %arg0, %[[MIN_INT_CST]], %[[X]]
func @test_minsi(%arg0 : i8) -> (i8, i8, i8, i8) {
  %maxIntCst = arith.constant 127 : i8
  %minIntCst = arith.constant -128 : i8
  %c0 = arith.constant 42 : i8
  %0 = arith.minsi %arg0, %arg0 : i8
  %1 = arith.minsi %arg0, %maxIntCst : i8
  %2 = arith.minsi %arg0, %minIntCst : i8
  %3 = arith.minsi %arg0, %c0 : i8
  return %0, %1, %2, %3: i8, i8, i8, i8
}

// -----

// CHECK-LABEL: test_minui
// CHECK: %[[C0:.+]] = arith.constant 42
// CHECK: %[[MIN_INT_CST:.+]] = arith.constant 0
// CHECK: %[[X:.+]] = arith.minui %arg0, %[[C0]]
// CHECK: return %arg0, %arg0, %[[MIN_INT_CST]], %[[X]]
func @test_minui(%arg0 : i8) -> (i8, i8, i8, i8) {
  %maxIntCst = arith.constant 255 : i8
  %minIntCst = arith.constant 0 : i8
  %c0 = arith.constant 42 : i8
  %0 = arith.minui %arg0, %arg0 : i8
  %1 = arith.minui %arg0, %maxIntCst : i8
  %2 = arith.minui %arg0, %minIntCst : i8
  %3 = arith.minui %arg0, %c0 : i8
  return %0, %1, %2, %3: i8, i8, i8, i8
}
