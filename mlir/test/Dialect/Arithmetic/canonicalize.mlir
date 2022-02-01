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

// CHECK-LABEL: @cmpOfExtSI
//  CHECK-NEXT:   return %arg0
func @cmpOfExtSI(%arg0: i1) -> i1 {
  %ext = arith.extsi %arg0 : i1 to i64
  %c0 = arith.constant 0 : i64
  %res = arith.cmpi ne, %ext, %c0 : i64
  return %res : i1
}

// CHECK-LABEL: @cmpOfExtUI
//  CHECK-NEXT:   return %arg0
func @cmpOfExtUI(%arg0: i1) -> i1 {
  %ext = arith.extui %arg0 : i1 to i64
  %c0 = arith.constant 0 : i64
  %res = arith.cmpi ne, %ext, %c0 : i64
  return %res : i1
}

// -----

// CHECK-LABEL: @extSIOfExtUI
//       CHECK:   %[[res:.+]] = arith.extui %arg0 : i1 to i64
//       CHECK:   return %[[res]]
func @extSIOfExtUI(%arg0: i1) -> i64 {
  %ext1 = arith.extui %arg0 : i1 to i8
  %ext2 = arith.extsi %ext1 : i8 to i64
  return %ext2 : i64
}

// CHECK-LABEL: @extUIOfExtUI
//       CHECK:   %[[res:.+]] = arith.extui %arg0 : i1 to i64
//       CHECK:   return %[[res]]
func @extUIOfExtUI(%arg0: i1) -> i64 {
  %ext1 = arith.extui %arg0 : i1 to i8
  %ext2 = arith.extui %ext1 : i8 to i64
  return %ext2 : i64
}

// CHECK-LABEL: @extSIOfExtSI
//       CHECK:   %[[res:.+]] = arith.extsi %arg0 : i1 to i64
//       CHECK:   return %[[res]]
func @extSIOfExtSI(%arg0: i1) -> i64 {
  %ext1 = arith.extsi %arg0 : i1 to i8
  %ext2 = arith.extsi %ext1 : i8 to i64
  return %ext2 : i64
}

// -----

// CHECK-LABEL: @andOfExtSI
//       CHECK:  %[[comb:.+]] = arith.andi %arg0, %arg1 : i8
//       CHECK:  %[[ext:.+]] = arith.extsi %[[comb]] : i8 to i64
//       CHECK:   return %[[ext]]
func @andOfExtSI(%arg0: i8, %arg1: i8) -> i64 {
  %ext0 = arith.extsi %arg0 : i8 to i64
  %ext1 = arith.extsi %arg1 : i8 to i64
  %res = arith.andi %ext0, %ext1 : i64
  return %res : i64
}

// CHECK-LABEL: @andOfExtUI
//       CHECK:  %[[comb:.+]] = arith.andi %arg0, %arg1 : i8
//       CHECK:  %[[ext:.+]] = arith.extui %[[comb]] : i8 to i64
//       CHECK:   return %[[ext]]
func @andOfExtUI(%arg0: i8, %arg1: i8) -> i64 {
  %ext0 = arith.extui %arg0 : i8 to i64
  %ext1 = arith.extui %arg1 : i8 to i64
  %res = arith.andi %ext0, %ext1 : i64
  return %res : i64
}

// CHECK-LABEL: @orOfExtSI
//       CHECK:  %[[comb:.+]] = arith.ori %arg0, %arg1 : i8
//       CHECK:  %[[ext:.+]] = arith.extsi %[[comb]] : i8 to i64
//       CHECK:   return %[[ext]]
func @orOfExtSI(%arg0: i8, %arg1: i8) -> i64 {
  %ext0 = arith.extsi %arg0 : i8 to i64
  %ext1 = arith.extsi %arg1 : i8 to i64
  %res = arith.ori %ext0, %ext1 : i64
  return %res : i64
}

// CHECK-LABEL: @orOfExtUI
//       CHECK:  %[[comb:.+]] = arith.ori %arg0, %arg1 : i8
//       CHECK:  %[[ext:.+]] = arith.extui %[[comb]] : i8 to i64
//       CHECK:   return %[[ext]]
func @orOfExtUI(%arg0: i8, %arg1: i8) -> i64 {
  %ext0 = arith.extui %arg0 : i8 to i64
  %ext1 = arith.extui %arg1 : i8 to i64
  %res = arith.ori %ext0, %ext1 : i64
  return %res : i64
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

// CHECK-LABEL: @truncTrunc
//       CHECK:   %[[cres:.+]] = arith.trunci %arg0 : i64 to i8
//       CHECK:   return %[[cres]]
func @truncTrunc(%arg0: i64) -> i8 {
  %tr1 = arith.trunci %arg0 : i64 to i32
  %tr2 = arith.trunci %tr1 : i32 to i8
  return %tr2 : i8
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

// CHECK-LABEL: @doubleAddSub1
//  CHECK-NEXT:   return %arg0
func @doubleAddSub1(%arg0: index, %arg1 : index) -> index {
  %sub = arith.subi %arg0, %arg1 : index
  %add = arith.addi %sub, %arg1 : index
  return %add : index
}

// CHECK-LABEL: @doubleAddSub2
//  CHECK-NEXT:   return %arg0
func @doubleAddSub2(%arg0: index, %arg1 : index) -> index {
  %sub = arith.subi %arg0, %arg1 : index
  %add = arith.addi %arg1, %sub : index
  return %add : index
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

// CHECK-LABEL: @xorxor(
//       CHECK-NOT: xori
//       CHECK:   return %arg0
func @xorxor(%cmp : i1) -> i1 {
  %true = arith.constant true
  %ncmp = arith.xori %cmp, %true : i1
  %nncmp = arith.xori %ncmp, %true : i1
  return %nncmp : i1
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

// -----

// CHECK-LABEL: @test_minf(
func @test_minf(%arg0 : f32) -> (f32, f32, f32) {
  // CHECK-DAG:   %[[C0:.+]] = arith.constant 0.0
  // CHECK-NEXT:  %[[X:.+]] = arith.minf %arg0, %[[C0]]
  // CHECK-NEXT:  return %[[X]], %arg0, %arg0
  %c0 = arith.constant 0.0 : f32
  %inf = arith.constant 0x7F800000 : f32
  %0 = arith.minf %c0, %arg0 : f32
  %1 = arith.minf %arg0, %arg0 : f32
  %2 = arith.minf %inf, %arg0 : f32
  return %0, %1, %2 : f32, f32, f32
}

// -----

// CHECK-LABEL: @test_maxf(
func @test_maxf(%arg0 : f32) -> (f32, f32, f32) {
  // CHECK-DAG:   %[[C0:.+]] = arith.constant
  // CHECK-NEXT:  %[[X:.+]] = arith.maxf %arg0, %[[C0]]
  // CHECK-NEXT:   return %[[X]], %arg0, %arg0
  %c0 = arith.constant 0.0 : f32
  %-inf = arith.constant 0xFF800000 : f32
  %0 = arith.maxf %c0, %arg0 : f32
  %1 = arith.maxf %arg0, %arg0 : f32
  %2 = arith.maxf %-inf, %arg0 : f32
  return %0, %1, %2 : f32, f32, f32
}

// -----

// CHECK-LABEL: @test_addf(
func @test_addf(%arg0 : f32) -> (f32, f32, f32, f32) {
  // CHECK-DAG:   %[[C2:.+]] = arith.constant 2.0
  // CHECK-DAG:   %[[C0:.+]] = arith.constant 0.0
  // CHECK-NEXT:  %[[X:.+]] = arith.addf %arg0, %[[C0]]
  // CHECK-NEXT:   return %[[X]], %arg0, %arg0, %[[C2]]
  %c0 = arith.constant 0.0 : f32
  %c-0 = arith.constant -0.0 : f32
  %c1 = arith.constant 1.0 : f32
  %0 = arith.addf %c0, %arg0 : f32
  %1 = arith.addf %arg0, %c-0 : f32
  %2 = arith.addf %c-0, %arg0 : f32
  %3 = arith.addf %c1, %c1 : f32
  return %0, %1, %2, %3 : f32, f32, f32, f32
}

// -----

// CHECK-LABEL: @test_subf(
func @test_subf(%arg0 : f16) -> (f16, f16, f16) {
  // CHECK-DAG:   %[[C1:.+]] = arith.constant -1.0
  // CHECK-DAG:   %[[C0:.+]] = arith.constant -0.0
  // CHECK-NEXT:  %[[X:.+]] = arith.subf %arg0, %[[C0]]
  // CHECK-NEXT:   return %arg0, %[[X]], %[[C1]]
  %c0 = arith.constant 0.0 : f16
  %c-0 = arith.constant -0.0 : f16
  %c1 = arith.constant 1.0 : f16
  %0 = arith.subf %arg0, %c0 : f16
  %1 = arith.subf %arg0, %c-0 : f16
  %2 = arith.subf %c0, %c1 : f16
  return %0, %1, %2 : f16, f16, f16
}

// -----

// CHECK-LABEL: @test_mulf(
func @test_mulf(%arg0 : f32) -> (f32, f32, f32, f32) {
  // CHECK-DAG:   %[[C2:.+]] = arith.constant 2.0
  // CHECK-DAG:   %[[C4:.+]] = arith.constant 4.0
  // CHECK-NEXT:  %[[X:.+]] = arith.mulf %arg0, %[[C2]]
  // CHECK-NEXT:  return %[[X]], %arg0, %arg0, %[[C4]]
  %c1 = arith.constant 1.0 : f32
  %c2 = arith.constant 2.0 : f32
  %0 = arith.mulf %c2, %arg0 : f32
  %1 = arith.mulf %arg0, %c1 : f32
  %2 = arith.mulf %c1, %arg0 : f32
  %3 = arith.mulf %c2, %c2 : f32
  return %0, %1, %2, %3 : f32, f32, f32, f32
}

// -----

// CHECK-LABEL: @test_divf(
func @test_divf(%arg0 : f64) -> (f64, f64) {
  // CHECK-NEXT:  %[[C5:.+]] = arith.constant 5.000000e-01
  // CHECK-NEXT:   return %arg0, %[[C5]]
  %c1 = arith.constant 1.0 : f64
  %c2 = arith.constant 2.0 : f64
  %0 = arith.divf %arg0, %c1 : f64
  %1 = arith.divf %c1, %c2 : f64
  return %0, %1 : f64, f64
}

// -----

// CHECK-LABEL: @test_cmpf(
func @test_cmpf(%arg0 : f32) -> (i1, i1, i1, i1) {
//   CHECK-DAG:   %[[T:.*]] = arith.constant true
//   CHECK-DAG:   %[[F:.*]] = arith.constant false
//       CHECK:   return %[[F]], %[[F]], %[[T]], %[[T]]
  %nan = arith.constant 0x7fffffff : f32
  %0 = arith.cmpf olt, %nan, %arg0 : f32
  %1 = arith.cmpf olt, %arg0, %nan : f32
  %2 = arith.cmpf ugt, %nan, %arg0 : f32
  %3 = arith.cmpf ugt, %arg0, %nan : f32
  return %0, %1, %2, %3 : i1, i1, i1, i1
}

// -----

// CHECK-LABEL: @constant_FPtoUI(
func @constant_FPtoUI() -> i32 {
  // CHECK: %[[C0:.+]] = arith.constant 2 : i32
  // CHECK: return %[[C0]]
  %c0 = arith.constant 2.0 : f32
  %res = arith.fptoui %c0 : f32 to i32
  return %res : i32
}

// -----
// CHECK-LABEL: @invalid_constant_FPtoUI(
func @invalid_constant_FPtoUI() -> i32 {
  // CHECK: %[[C0:.+]] = arith.constant -2.000000e+00 : f32
  // CHECK: %[[C1:.+]] = arith.fptoui %[[C0]] : f32 to i32
  // CHECK: return %[[C1]]
  %c0 = arith.constant -2.0 : f32
  %res = arith.fptoui %c0 : f32 to i32
  return %res : i32
}

// -----
// CHECK-LABEL: @constant_FPtoSI(
func @constant_FPtoSI() -> i32 {
  // CHECK: %[[C0:.+]] = arith.constant -2 : i32
  // CHECK: return %[[C0]]
  %c0 = arith.constant -2.0 : f32
  %res = arith.fptosi %c0 : f32 to i32
  return %res : i32
}

// -----
// CHECK-LABEL: @invalid_constant_FPtoSI(
func @invalid_constant_FPtoSI() -> i8 {
  // CHECK: %[[C0:.+]] = arith.constant 2.000000e+10 : f32
  // CHECK: %[[C1:.+]] = arith.fptosi %[[C0]] : f32 to i8
  // CHECK: return %[[C1]]
  %c0 = arith.constant 2.0e10 : f32
  %res = arith.fptosi %c0 : f32 to i8
  return %res : i8
}

// CHECK-LABEL: @constant_SItoFP(
func @constant_SItoFP() -> f32 {
  // CHECK: %[[C0:.+]] = arith.constant -2.000000e+00 : f32
  // CHECK: return %[[C0]]
  %c0 = arith.constant -2 : i32
  %res = arith.sitofp %c0 : i32 to f32
  return %res : f32
}

// -----
// CHECK-LABEL: @constant_UItoFP(
func @constant_UItoFP() -> f32 {
  // CHECK: %[[C0:.+]] = arith.constant 2.000000e+00 : f32
  // CHECK: return %[[C0]]
  %c0 = arith.constant 2 : i32
  %res = arith.sitofp %c0 : i32 to f32
  return %res : f32
}
