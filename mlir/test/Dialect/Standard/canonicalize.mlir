// RUN: mlir-opt %s -canonicalize --split-input-file | FileCheck %s

// Test case: Folding of comparisons with equal operands.
// CHECK-LABEL: @cmpi_equal_operands
//   CHECK-DAG:   %[[T:.*]] = constant true
//   CHECK-DAG:   %[[F:.*]] = constant false
//       CHECK:   return %[[T]], %[[T]], %[[T]], %[[T]], %[[T]],
//  CHECK-SAME:          %[[F]], %[[F]], %[[F]], %[[F]], %[[F]]
func @cmpi_equal_operands(%arg0: i64)
    -> (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1) {
  %0 = cmpi eq, %arg0, %arg0 : i64
  %1 = cmpi sle, %arg0, %arg0 : i64
  %2 = cmpi sge, %arg0, %arg0 : i64
  %3 = cmpi ule, %arg0, %arg0 : i64
  %4 = cmpi uge, %arg0, %arg0 : i64
  %5 = cmpi ne, %arg0, %arg0 : i64
  %6 = cmpi slt, %arg0, %arg0 : i64
  %7 = cmpi sgt, %arg0, %arg0 : i64
  %8 = cmpi ult, %arg0, %arg0 : i64
  %9 = cmpi ugt, %arg0, %arg0 : i64
  return %0, %1, %2, %3, %4, %5, %6, %7, %8, %9
      : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
}

// -----

// CHECK-LABEL: @select_same_val
//       CHECK:   return %arg1
func @select_same_val(%arg0: i1, %arg1: i64) -> i64 {
  %0 = select %arg0, %arg1, %arg1 : i64
  return %0 : i64
}

// -----

// CHECK-LABEL: @select_cmp_eq_select
//       CHECK:   return %arg1
func @select_cmp_eq_select(%arg0: i64, %arg1: i64) -> i64 {
  %0 = cmpi eq, %arg0, %arg1 : i64
  %1 = select %0, %arg0, %arg1 : i64
  return %1 : i64
}

// -----

// CHECK-LABEL: @select_cmp_ne_select
//       CHECK:   return %arg0
func @select_cmp_ne_select(%arg0: i64, %arg1: i64) -> i64 {
  %0 = cmpi ne, %arg0, %arg1 : i64
  %1 = select %0, %arg0, %arg1 : i64
  return %1 : i64
}

// -----

// CHECK-LABEL: @indexCastOfSignExtend
//       CHECK:   %[[res:.+]] = index_cast %arg0 : i8 to index
//       CHECK:   return %[[res]]
func @indexCastOfSignExtend(%arg0: i8) -> index {
  %ext = sexti %arg0 : i8 to i16
  %idx = index_cast %ext : i16 to index
  return %idx : index
}

// CHECK-LABEL: @signExtendConstant
//       CHECK:   %[[cres:.+]] = constant -2 : i16
//       CHECK:   return %[[cres]]
func @signExtendConstant() -> i16 {
  %c-2 = constant -2 : i8
  %ext = sexti %c-2 : i8 to i16
  return %ext : i16
}

// CHECK-LABEL: @truncConstant
//       CHECK:   %[[cres:.+]] = constant -2 : i16
//       CHECK:   return %[[cres]]
func @truncConstant(%arg0: i8) -> i16 {
  %c-2 = constant -2 : i32
  %tr = trunci %c-2 : i32 to i16
  return %tr : i16
}

// CHECK-LABEL: @tripleAddAdd
//       CHECK:   %[[cres:.+]] = constant 59 : index 
//       CHECK:   %[[add:.+]] = addi %arg0, %[[cres]] : index 
//       CHECK:   return %[[add]]
func @tripleAddAdd(%arg0: index) -> index {
  %c17 = constant 17 : index
  %c42 = constant 42 : index
  %add1 = addi %c17, %arg0 : index
  %add2 = addi %c42, %add1 : index
  return %add2 : index
}

// CHECK-LABEL: @tripleAddSub0
//       CHECK:   %[[cres:.+]] = constant 59 : index 
//       CHECK:   %[[add:.+]] = subi %[[cres]], %arg0 : index 
//       CHECK:   return %[[add]]
func @tripleAddSub0(%arg0: index) -> index {
  %c17 = constant 17 : index
  %c42 = constant 42 : index
  %add1 = subi %c17, %arg0 : index
  %add2 = addi %c42, %add1 : index
  return %add2 : index
}

// CHECK-LABEL: @tripleAddSub1
//       CHECK:   %[[cres:.+]] = constant 25 : index 
//       CHECK:   %[[add:.+]] = addi %arg0, %[[cres]] : index 
//       CHECK:   return %[[add]]
func @tripleAddSub1(%arg0: index) -> index {
  %c17 = constant 17 : index
  %c42 = constant 42 : index
  %add1 = subi %arg0, %c17 : index
  %add2 = addi %c42, %add1 : index
  return %add2 : index
}

// CHECK-LABEL: @tripleSubAdd0
//       CHECK:   %[[cres:.+]] = constant 25 : index 
//       CHECK:   %[[add:.+]] = subi %[[cres]], %arg0 : index 
//       CHECK:   return %[[add]]
func @tripleSubAdd0(%arg0: index) -> index {
  %c17 = constant 17 : index
  %c42 = constant 42 : index
  %add1 = addi %c17, %arg0 : index
  %add2 = subi %c42, %add1 : index
  return %add2 : index
}

// CHECK-LABEL: @tripleSubAdd1
//       CHECK:   %[[cres:.+]] = constant -25 : index 
//       CHECK:   %[[add:.+]] = addi %arg0, %[[cres]] : index 
//       CHECK:   return %[[add]]
func @tripleSubAdd1(%arg0: index) -> index {
  %c17 = constant 17 : index
  %c42 = constant 42 : index
  %add1 = addi %c17, %arg0 : index
  %add2 = subi %add1, %c42 : index
  return %add2 : index
}

// CHECK-LABEL: @tripleSubSub0
//       CHECK:   %[[cres:.+]] = constant 25 : index 
//       CHECK:   %[[add:.+]] = addi %arg0, %[[cres]] : index 
//       CHECK:   return %[[add]]
func @tripleSubSub0(%arg0: index) -> index {
  %c17 = constant 17 : index
  %c42 = constant 42 : index
  %add1 = subi %c17, %arg0 : index
  %add2 = subi %c42, %add1 : index
  return %add2 : index
}

// CHECK-LABEL: @tripleSubSub1
//       CHECK:   %[[cres:.+]] = constant -25 : index 
//       CHECK:   %[[add:.+]] = subi %[[cres]], %arg0 : index 
//       CHECK:   return %[[add]]
func @tripleSubSub1(%arg0: index) -> index {
  %c17 = constant 17 : index
  %c42 = constant 42 : index
  %add1 = subi %c17, %arg0 : index
  %add2 = subi %add1, %c42 : index
  return %add2 : index
}

// CHECK-LABEL: @tripleSubSub2
//       CHECK:   %[[cres:.+]] = constant 59 : index 
//       CHECK:   %[[add:.+]] = subi %[[cres]], %arg0 : index 
//       CHECK:   return %[[add]]
func @tripleSubSub2(%arg0: index) -> index {
  %c17 = constant 17 : index
  %c42 = constant 42 : index
  %add1 = subi %arg0, %c17 : index
  %add2 = subi %c42, %add1 : index
  return %add2 : index
}

// CHECK-LABEL: @tripleSubSub3
//       CHECK:   %[[cres:.+]] = constant 59 : index 
//       CHECK:   %[[add:.+]] = subi %arg0, %[[cres]] : index 
//       CHECK:   return %[[add]]
func @tripleSubSub3(%arg0: index) -> index {
  %c17 = constant 17 : index
  %c42 = constant 42 : index
  %add1 = subi %arg0, %c17 : index
  %add2 = subi %add1, %c42 : index
  return %add2 : index
}

// CHECK-LABEL: @notCmpEQ
//       CHECK:   %[[cres:.+]] = cmpi ne, %arg0, %arg1 : i8
//       CHECK:   return %[[cres]]
func @notCmpEQ(%arg0: i8, %arg1: i8) -> i1 {
  %true = constant true
  %cmp = cmpi "eq", %arg0, %arg1 : i8
  %ncmp = xor %cmp, %true : i1
  return %ncmp : i1
}

// CHECK-LABEL: @notCmpEQ2
//       CHECK:   %[[cres:.+]] = cmpi ne, %arg0, %arg1 : i8
//       CHECK:   return %[[cres]]
func @notCmpEQ2(%arg0: i8, %arg1: i8) -> i1 {
  %true = constant true
  %cmp = cmpi "eq", %arg0, %arg1 : i8
  %ncmp = xor %true, %cmp : i1
  return %ncmp : i1
}

// CHECK-LABEL: @notCmpNE
//       CHECK:   %[[cres:.+]] = cmpi eq, %arg0, %arg1 : i8
//       CHECK:   return %[[cres]]
func @notCmpNE(%arg0: i8, %arg1: i8) -> i1 {
  %true = constant true
  %cmp = cmpi "ne", %arg0, %arg1 : i8
  %ncmp = xor %cmp, %true : i1
  return %ncmp : i1
}

// CHECK-LABEL: @notCmpSLT
//       CHECK:   %[[cres:.+]] = cmpi sge, %arg0, %arg1 : i8
//       CHECK:   return %[[cres]]
func @notCmpSLT(%arg0: i8, %arg1: i8) -> i1 {
  %true = constant true
  %cmp = cmpi "slt", %arg0, %arg1 : i8
  %ncmp = xor %cmp, %true : i1
  return %ncmp : i1
}

// CHECK-LABEL: @notCmpSLE
//       CHECK:   %[[cres:.+]] = cmpi sgt, %arg0, %arg1 : i8
//       CHECK:   return %[[cres]]
func @notCmpSLE(%arg0: i8, %arg1: i8) -> i1 {
  %true = constant true
  %cmp = cmpi "sle", %arg0, %arg1 : i8
  %ncmp = xor %cmp, %true : i1
  return %ncmp : i1
}

// CHECK-LABEL: @notCmpSGT
//       CHECK:   %[[cres:.+]] = cmpi sle, %arg0, %arg1 : i8
//       CHECK:   return %[[cres]]
func @notCmpSGT(%arg0: i8, %arg1: i8) -> i1 {
  %true = constant true
  %cmp = cmpi "sgt", %arg0, %arg1 : i8
  %ncmp = xor %cmp, %true : i1
  return %ncmp : i1
}

// CHECK-LABEL: @notCmpSGE
//       CHECK:   %[[cres:.+]] = cmpi slt, %arg0, %arg1 : i8
//       CHECK:   return %[[cres]]
func @notCmpSGE(%arg0: i8, %arg1: i8) -> i1 {
  %true = constant true
  %cmp = cmpi "sge", %arg0, %arg1 : i8
  %ncmp = xor %cmp, %true : i1
  return %ncmp : i1
}

// CHECK-LABEL: @notCmpULT
//       CHECK:   %[[cres:.+]] = cmpi uge, %arg0, %arg1 : i8
//       CHECK:   return %[[cres]]
func @notCmpULT(%arg0: i8, %arg1: i8) -> i1 {
  %true = constant true
  %cmp = cmpi "ult", %arg0, %arg1 : i8
  %ncmp = xor %cmp, %true : i1
  return %ncmp : i1
}

// CHECK-LABEL: @notCmpULE
//       CHECK:   %[[cres:.+]] = cmpi ugt, %arg0, %arg1 : i8
//       CHECK:   return %[[cres]]
func @notCmpULE(%arg0: i8, %arg1: i8) -> i1 {
  %true = constant true
  %cmp = cmpi "ule", %arg0, %arg1 : i8
  %ncmp = xor %cmp, %true : i1
  return %ncmp : i1
}

// CHECK-LABEL: @notCmpUGT
//       CHECK:   %[[cres:.+]] = cmpi ule, %arg0, %arg1 : i8
//       CHECK:   return %[[cres]]
func @notCmpUGT(%arg0: i8, %arg1: i8) -> i1 {
  %true = constant true
  %cmp = cmpi "ugt", %arg0, %arg1 : i8
  %ncmp = xor %cmp, %true : i1
  return %ncmp : i1
}

// CHECK-LABEL: @notCmpUGE
//       CHECK:   %[[cres:.+]] = cmpi ult, %arg0, %arg1 : i8
//       CHECK:   return %[[cres]]
func @notCmpUGE(%arg0: i8, %arg1: i8) -> i1 {
  %true = constant true
  %cmp = cmpi "uge", %arg0, %arg1 : i8
  %ncmp = xor %cmp, %true : i1
  return %ncmp : i1
}

// -----

// CHECK-LABEL: @branchCondProp
//       CHECK:       %[[trueval:.+]] = constant true
//       CHECK:       %[[falseval:.+]] = constant false
//       CHECK:       "test.consumer1"(%[[trueval]]) : (i1) -> ()
//       CHECK:       "test.consumer2"(%[[falseval]]) : (i1) -> ()
func @branchCondProp(%arg0: i1) {
  cond_br %arg0, ^trueB, ^falseB

^trueB:
  "test.consumer1"(%arg0) : (i1) -> ()
  br ^exit

^falseB:
  "test.consumer2"(%arg0) : (i1) -> ()
  br ^exit

^exit:
  return
}

// -----

// CHECK-LABEL: @selToNot
//       CHECK:       %[[trueval:.+]] = constant true
//       CHECK:       %{{.+}} = xor %arg0, %[[trueval]] : i1
func @selToNot(%arg0: i1) -> i1 {
  %true = constant true
  %false = constant false
  %res = select %arg0, %false, %true : i1
  return %res : i1
}

// -----

// CHECK-LABEL: @bitcastSameType(
// CHECK-SAME: %[[ARG:[a-zA-Z0-9_]*]]
func @bitcastSameType(%arg : f32) -> f32 {
  // CHECK: return %[[ARG]]
  %res = bitcast %arg : f32 to f32
  return %res : f32
}

// -----

// CHECK-LABEL: @bitcastConstantFPtoI(
func @bitcastConstantFPtoI() -> i32 {
  // CHECK: %[[C0:.+]] = constant 0 : i32
  // CHECK: return %[[C0]]
  %c0 = constant 0.0 : f32
  %res = bitcast %c0 : f32 to i32
  return %res : i32
}

// -----

// CHECK-LABEL: @bitcastConstantItoFP(
func @bitcastConstantItoFP() -> f32 {
  // CHECK: %[[C0:.+]] = constant 0.0{{.*}} : f32
  // CHECK: return %[[C0]]
  %c0 = constant 0 : i32
  %res = bitcast %c0 : i32 to f32
  return %res : f32
}

// -----

// CHECK-LABEL: @bitcastConstantFPtoFP(
func @bitcastConstantFPtoFP() -> f16 {
  // CHECK: %[[C0:.+]] = constant 0.0{{.*}} : f16
  // CHECK: return %[[C0]]
  %c0 = constant 0.0 : bf16
  %res = bitcast %c0 : bf16 to f16
  return %res : f16
}

// -----

// CHECK-LABEL: @bitcastConstantVecFPtoI(
func @bitcastConstantVecFPtoI() -> vector<3xf32> {
  // CHECK: %[[C0:.+]] = constant dense<0.0{{.*}}> : vector<3xf32>
  // CHECK: return %[[C0]]
  %c0 = constant dense<0> : vector<3xi32>
  %res = bitcast %c0 : vector<3xi32> to vector<3xf32>
  return %res : vector<3xf32>
}

// -----

// CHECK-LABEL: @bitcastConstantVecItoFP(
func @bitcastConstantVecItoFP() -> vector<3xi32> {
  // CHECK: %[[C0:.+]] = constant dense<0> : vector<3xi32>
  // CHECK: return %[[C0]]
  %c0 = constant dense<0.0> : vector<3xf32>
  %res = bitcast %c0 : vector<3xf32> to vector<3xi32>
  return %res : vector<3xi32>
}

// -----

// CHECK-LABEL: @bitcastConstantVecFPtoFP(
func @bitcastConstantVecFPtoFP() -> vector<3xbf16> {
  // CHECK: %[[C0:.+]] = constant dense<0.0{{.*}}> : vector<3xbf16>
  // CHECK: return %[[C0]]
  %c0 = constant dense<0.0> : vector<3xf16>
  %res = bitcast %c0 : vector<3xf16> to vector<3xbf16>
  return %res : vector<3xbf16>
}

// -----

// CHECK-LABEL: @bitcastBackAndForth(
// CHECK-SAME: %[[ARG:[a-zA-Z0-9_]*]]
func @bitcastBackAndForth(%arg : i32) -> i32 {
  // CHECK: return %[[ARG]]
  %f = bitcast %arg : i32 to f32
  %res = bitcast %f : f32 to i32
  return %res : i32
}

// -----

// CHECK-LABEL: @bitcastOfBitcast(
// CHECK-SAME: %[[ARG:[a-zA-Z0-9_]*]]
func @bitcastOfBitcast(%arg : i16) -> i16 {
  // CHECK: return %[[ARG]]
  %f = bitcast %arg : i16 to f16
  %bf = bitcast %f : f16 to bf16
  %res = bitcast %bf : bf16 to i16
  return %res : i16
}
