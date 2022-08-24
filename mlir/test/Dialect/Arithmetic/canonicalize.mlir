// RUN: mlir-opt %s -canonicalize --split-input-file | FileCheck %s

// CHECK-LABEL: @select_same_val
//       CHECK:   return %arg1
func.func @select_same_val(%arg0: i1, %arg1: i64) -> i64 {
  %0 = arith.select %arg0, %arg1, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: @select_cmp_eq_select
//       CHECK:   return %arg1
func.func @select_cmp_eq_select(%arg0: i64, %arg1: i64) -> i64 {
  %0 = arith.cmpi eq, %arg0, %arg1 : i64
  %1 = arith.select %0, %arg0, %arg1 : i64
  return %1 : i64
}

// CHECK-LABEL: @select_cmp_ne_select
//       CHECK:   return %arg0
func.func @select_cmp_ne_select(%arg0: i64, %arg1: i64) -> i64 {
  %0 = arith.cmpi ne, %arg0, %arg1 : i64
  %1 = arith.select %0, %arg0, %arg1 : i64
  return %1 : i64
}

// CHECK-LABEL: @select_extui
//       CHECK:   %[[res:.+]] = arith.extui %arg0 : i1 to i64
//       CHECK:   return %[[res]]
func.func @select_extui(%arg0: i1) -> i64 {
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %res = arith.select %arg0, %c1_i64, %c0_i64 : i64
  return %res : i64
}

// CHECK-LABEL: @select_extui2
// CHECK-DAG:  %true = arith.constant true
// CHECK-DAG:  %[[xor:.+]] = arith.xori %arg0, %true : i1
// CHECK-DAG:  %[[res:.+]] = arith.extui %[[xor]] : i1 to i64
//       CHECK:   return %[[res]]
func.func @select_extui2(%arg0: i1) -> i64 {
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %res = arith.select %arg0, %c0_i64, %c1_i64 : i64
  return %res : i64
}

// CHECK-LABEL: @select_extui_i1
//  CHECK-NEXT:   return %arg0
func.func @select_extui_i1(%arg0: i1) -> i1 {
  %c0_i1 = arith.constant false
  %c1_i1 = arith.constant true
  %res = arith.select %arg0, %c1_i1, %c0_i1 : i1
  return %res : i1
}

// CHECK-LABEL: @selToNot
//       CHECK:       %[[trueval:.+]] = arith.constant true
//       CHECK:       %[[res:.+]] = arith.xori %arg0, %[[trueval]] : i1
//       CHECK:   return %[[res]]
func.func @selToNot(%arg0: i1) -> i1 {
  %true = arith.constant true
  %false = arith.constant false
  %res = arith.select %arg0, %false, %true : i1
  return %res : i1
}

// CHECK-LABEL: @selToArith
//       CHECK-NEXT:       %[[trueval:.+]] = arith.constant true
//       CHECK-NEXT:       %[[notcmp:.+]] = arith.xori %arg0, %[[trueval]] : i1
//       CHECK-NEXT:       %[[condtrue:.+]] = arith.andi %arg0, %arg1 : i1
//       CHECK-NEXT:       %[[condfalse:.+]] = arith.andi %[[notcmp]], %arg2 : i1
//       CHECK-NEXT:       %[[res:.+]] = arith.ori %[[condtrue]], %[[condfalse]] : i1
//       CHECK:   return %[[res]]
func.func @selToArith(%arg0: i1, %arg1 : i1, %arg2 : i1) -> i1 {
  %res = arith.select %arg0, %arg1, %arg2 : i1
  return %res : i1
}

// Test case: Folding of comparisons with equal operands.
// CHECK-LABEL: @cmpi_equal_operands
//   CHECK-DAG:   %[[T:.*]] = arith.constant true
//   CHECK-DAG:   %[[F:.*]] = arith.constant false
//       CHECK:   return %[[T]], %[[T]], %[[T]], %[[T]], %[[T]],
//  CHECK-SAME:          %[[F]], %[[F]], %[[F]], %[[F]], %[[F]]
func.func @cmpi_equal_operands(%arg0: i64)
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
func.func @cmpi_equal_vector_operands(%arg0: vector<1x8xi64>)
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

// Test case: Move constant to the right side.
// CHECK-LABEL: @cmpi_const_right(
//  CHECK-SAME: %[[ARG:.*]]:
//       CHECK:   %[[C:.*]] = arith.constant 1 : i64
//       CHECK:   %[[R0:.*]] = arith.cmpi eq, %[[ARG]], %[[C]] : i64
//       CHECK:   %[[R1:.*]] = arith.cmpi sge, %[[ARG]], %[[C]] : i64
//       CHECK:   %[[R2:.*]] = arith.cmpi sle, %[[ARG]], %[[C]] : i64
//       CHECK:   %[[R3:.*]] = arith.cmpi uge, %[[ARG]], %[[C]] : i64
//       CHECK:   %[[R4:.*]] = arith.cmpi ule, %[[ARG]], %[[C]] : i64
//       CHECK:   %[[R5:.*]] = arith.cmpi ne, %[[ARG]], %[[C]] : i64
//       CHECK:   %[[R6:.*]] = arith.cmpi sgt, %[[ARG]], %[[C]] : i64
//       CHECK:   %[[R7:.*]] = arith.cmpi slt, %[[ARG]], %[[C]] : i64
//       CHECK:   %[[R8:.*]] = arith.cmpi ugt, %[[ARG]], %[[C]] : i64
//       CHECK:   %[[R9:.*]] = arith.cmpi ult, %[[ARG]], %[[C]] : i64
//       CHECK:   return %[[R0]], %[[R1]], %[[R2]], %[[R3]], %[[R4]],
//  CHECK-SAME:          %[[R5]], %[[R6]], %[[R7]], %[[R8]], %[[R9]]
func.func @cmpi_const_right(%arg0: i64)
    -> (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1) {
  %c1 = arith.constant 1 : i64
  %0 = arith.cmpi eq, %c1, %arg0 : i64
  %1 = arith.cmpi sle, %c1, %arg0 : i64
  %2 = arith.cmpi sge, %c1, %arg0 : i64
  %3 = arith.cmpi ule, %c1, %arg0 : i64
  %4 = arith.cmpi uge, %c1, %arg0 : i64
  %5 = arith.cmpi ne, %c1, %arg0 : i64
  %6 = arith.cmpi slt, %c1, %arg0 : i64
  %7 = arith.cmpi sgt, %c1, %arg0 : i64
  %8 = arith.cmpi ult, %c1, %arg0 : i64
  %9 = arith.cmpi ugt, %c1, %arg0 : i64
  return %0, %1, %2, %3, %4, %5, %6, %7, %8, %9
      : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
}

// -----

// CHECK-LABEL: @cmpOfExtSI
//  CHECK-NEXT:   return %arg0
func.func @cmpOfExtSI(%arg0: i1) -> i1 {
  %ext = arith.extsi %arg0 : i1 to i64
  %c0 = arith.constant 0 : i64
  %res = arith.cmpi ne, %ext, %c0 : i64
  return %res : i1
}

// CHECK-LABEL: @cmpOfExtUI
//  CHECK-NEXT:   return %arg0
func.func @cmpOfExtUI(%arg0: i1) -> i1 {
  %ext = arith.extui %arg0 : i1 to i64
  %c0 = arith.constant 0 : i64
  %res = arith.cmpi ne, %ext, %c0 : i64
  return %res : i1
}

// -----

// CHECK-LABEL: @extSIOfExtUI
//       CHECK:   %[[res:.+]] = arith.extui %arg0 : i1 to i64
//       CHECK:   return %[[res]]
func.func @extSIOfExtUI(%arg0: i1) -> i64 {
  %ext1 = arith.extui %arg0 : i1 to i8
  %ext2 = arith.extsi %ext1 : i8 to i64
  return %ext2 : i64
}

// CHECK-LABEL: @extUIOfExtUI
//       CHECK:   %[[res:.+]] = arith.extui %arg0 : i1 to i64
//       CHECK:   return %[[res]]
func.func @extUIOfExtUI(%arg0: i1) -> i64 {
  %ext1 = arith.extui %arg0 : i1 to i8
  %ext2 = arith.extui %ext1 : i8 to i64
  return %ext2 : i64
}

// CHECK-LABEL: @extSIOfExtSI
//       CHECK:   %[[res:.+]] = arith.extsi %arg0 : i1 to i64
//       CHECK:   return %[[res]]
func.func @extSIOfExtSI(%arg0: i1) -> i64 {
  %ext1 = arith.extsi %arg0 : i1 to i8
  %ext2 = arith.extsi %ext1 : i8 to i64
  return %ext2 : i64
}

// -----

// CHECK-LABEL: @cmpIExtSINE
//       CHECK:  %[[comb:.+]] = arith.cmpi ne, %arg0, %arg1 : i8
//       CHECK:   return %[[comb]]
func.func @cmpIExtSINE(%arg0: i8, %arg1: i8) -> i1 {
  %ext0 = arith.extsi %arg0 : i8 to i64
  %ext1 = arith.extsi %arg1 : i8 to i64
  %res = arith.cmpi ne, %ext0, %ext1 : i64
  return %res : i1
}

// CHECK-LABEL: @cmpIExtSIEQ
//       CHECK:  %[[comb:.+]] = arith.cmpi eq, %arg0, %arg1 : i8
//       CHECK:   return %[[comb]]
func.func @cmpIExtSIEQ(%arg0: i8, %arg1: i8) -> i1 {
  %ext0 = arith.extsi %arg0 : i8 to i64
  %ext1 = arith.extsi %arg1 : i8 to i64
  %res = arith.cmpi eq, %ext0, %ext1 : i64
  return %res : i1
}

// CHECK-LABEL: @cmpIExtUINE
//       CHECK:  %[[comb:.+]] = arith.cmpi ne, %arg0, %arg1 : i8
//       CHECK:   return %[[comb]]
func.func @cmpIExtUINE(%arg0: i8, %arg1: i8) -> i1 {
  %ext0 = arith.extui %arg0 : i8 to i64
  %ext1 = arith.extui %arg1 : i8 to i64
  %res = arith.cmpi ne, %ext0, %ext1 : i64
  return %res : i1
}

// CHECK-LABEL: @cmpIExtUIEQ
//       CHECK:  %[[comb:.+]] = arith.cmpi eq, %arg0, %arg1 : i8
//       CHECK:   return %[[comb]]
func.func @cmpIExtUIEQ(%arg0: i8, %arg1: i8) -> i1 {
  %ext0 = arith.extui %arg0 : i8 to i64
  %ext1 = arith.extui %arg1 : i8 to i64
  %res = arith.cmpi eq, %ext0, %ext1 : i64
  return %res : i1
}

// -----

// CHECK-LABEL: @andOfExtSI
//       CHECK:  %[[comb:.+]] = arith.andi %arg0, %arg1 : i8
//       CHECK:  %[[ext:.+]] = arith.extsi %[[comb]] : i8 to i64
//       CHECK:   return %[[ext]]
func.func @andOfExtSI(%arg0: i8, %arg1: i8) -> i64 {
  %ext0 = arith.extsi %arg0 : i8 to i64
  %ext1 = arith.extsi %arg1 : i8 to i64
  %res = arith.andi %ext0, %ext1 : i64
  return %res : i64
}

// CHECK-LABEL: @andOfExtUI
//       CHECK:  %[[comb:.+]] = arith.andi %arg0, %arg1 : i8
//       CHECK:  %[[ext:.+]] = arith.extui %[[comb]] : i8 to i64
//       CHECK:   return %[[ext]]
func.func @andOfExtUI(%arg0: i8, %arg1: i8) -> i64 {
  %ext0 = arith.extui %arg0 : i8 to i64
  %ext1 = arith.extui %arg1 : i8 to i64
  %res = arith.andi %ext0, %ext1 : i64
  return %res : i64
}

// CHECK-LABEL: @orOfExtSI
//       CHECK:  %[[comb:.+]] = arith.ori %arg0, %arg1 : i8
//       CHECK:  %[[ext:.+]] = arith.extsi %[[comb]] : i8 to i64
//       CHECK:   return %[[ext]]
func.func @orOfExtSI(%arg0: i8, %arg1: i8) -> i64 {
  %ext0 = arith.extsi %arg0 : i8 to i64
  %ext1 = arith.extsi %arg1 : i8 to i64
  %res = arith.ori %ext0, %ext1 : i64
  return %res : i64
}

// CHECK-LABEL: @orOfExtUI
//       CHECK:  %[[comb:.+]] = arith.ori %arg0, %arg1 : i8
//       CHECK:  %[[ext:.+]] = arith.extui %[[comb]] : i8 to i64
//       CHECK:   return %[[ext]]
func.func @orOfExtUI(%arg0: i8, %arg1: i8) -> i64 {
  %ext0 = arith.extui %arg0 : i8 to i64
  %ext1 = arith.extui %arg1 : i8 to i64
  %res = arith.ori %ext0, %ext1 : i64
  return %res : i64
}

// -----

// CHECK-LABEL: @indexCastOfSignExtend
//       CHECK:   %[[res:.+]] = arith.index_cast %arg0 : i8 to index
//       CHECK:   return %[[res]]
func.func @indexCastOfSignExtend(%arg0: i8) -> index {
  %ext = arith.extsi %arg0 : i8 to i16
  %idx = arith.index_cast %ext : i16 to index
  return %idx : index
}

// CHECK-LABEL: @signExtendConstant
//       CHECK:   %[[cres:.+]] = arith.constant -2 : i16
//       CHECK:   return %[[cres]]
func.func @signExtendConstant() -> i16 {
  %c-2 = arith.constant -2 : i8
  %ext = arith.extsi %c-2 : i8 to i16
  return %ext : i16
}

// CHECK-LABEL: @signExtendConstantSplat
//       CHECK:   %[[cres:.+]] = arith.constant dense<-2> : vector<4xi16>
//       CHECK:   return %[[cres]]
func.func @signExtendConstantSplat() -> vector<4xi16> {
  %c-2 = arith.constant -2 : i8
  %splat = vector.splat %c-2 : vector<4xi8>
  %ext = arith.extsi %splat : vector<4xi8> to vector<4xi16>
  return %ext : vector<4xi16>
}

// CHECK-LABEL: @signExtendConstantVector
//       CHECK:   %[[cres:.+]] = arith.constant dense<[1, 3, 5, 7]> : vector<4xi16>
//       CHECK:   return %[[cres]]
func.func @signExtendConstantVector() -> vector<4xi16> {
  %vector = arith.constant dense<[1, 3, 5, 7]> : vector<4xi8>
  %ext = arith.extsi %vector : vector<4xi8> to vector<4xi16>
  return %ext : vector<4xi16>
}

// CHECK-LABEL: @unsignedExtendConstant
//       CHECK:   %[[cres:.+]] = arith.constant 2 : i16
//       CHECK:   return %[[cres]]
func.func @unsignedExtendConstant() -> i16 {
  %c2 = arith.constant 2 : i8
  %ext = arith.extui %c2 : i8 to i16
  return %ext : i16
}

// CHECK-LABEL: @unsignedExtendConstantSplat
//       CHECK:   %[[cres:.+]] = arith.constant dense<2> : vector<4xi16>
//       CHECK:   return %[[cres]]
func.func @unsignedExtendConstantSplat() -> vector<4xi16> {
  %c2 = arith.constant 2 : i8
  %splat = vector.splat %c2 : vector<4xi8>
  %ext = arith.extui %splat : vector<4xi8> to vector<4xi16>
  return %ext : vector<4xi16>
}

// CHECK-LABEL: @unsignedExtendConstantVector
//       CHECK:   %[[cres:.+]] = arith.constant dense<[1, 3, 5, 7]> : vector<4xi16>
//       CHECK:   return %[[cres]]
func.func @unsignedExtendConstantVector() -> vector<4xi16> {
  %vector = arith.constant dense<[1, 3, 5, 7]> : vector<4xi8>
  %ext = arith.extui %vector : vector<4xi8> to vector<4xi16>
  return %ext : vector<4xi16>
}

// CHECK-LABEL: @truncConstant
//       CHECK:   %[[cres:.+]] = arith.constant -2 : i16
//       CHECK:   return %[[cres]]
func.func @truncConstant(%arg0: i8) -> i16 {
  %c-2 = arith.constant -2 : i32
  %tr = arith.trunci %c-2 : i32 to i16
  return %tr : i16
}

// CHECK-LABEL: @truncConstantSplat
//       CHECK:   %[[cres:.+]] = arith.constant dense<-2> : vector<4xi8>
//       CHECK:   return %[[cres]]
func.func @truncConstantSplat() -> vector<4xi8> {
  %c-2 = arith.constant -2 : i16
  %splat = vector.splat %c-2 : vector<4xi16>
  %trunc = arith.trunci %splat : vector<4xi16> to vector<4xi8>
  return %trunc : vector<4xi8>
}

// CHECK-LABEL: @truncConstantVector
//       CHECK:   %[[cres:.+]] = arith.constant dense<[1, 3, 5, 7]> : vector<4xi8>
//       CHECK:   return %[[cres]]
func.func @truncConstantVector() -> vector<4xi8> {
  %vector = arith.constant dense<[1, 3, 5, 7]> : vector<4xi16>
  %trunc = arith.trunci %vector : vector<4xi16> to vector<4xi8>
  return %trunc : vector<4xi8>
}

// CHECK-LABEL: @truncTrunc
//       CHECK:   %[[cres:.+]] = arith.trunci %arg0 : i64 to i8
//       CHECK:   return %[[cres]]
func.func @truncTrunc(%arg0: i64) -> i8 {
  %tr1 = arith.trunci %arg0 : i64 to i32
  %tr2 = arith.trunci %tr1 : i32 to i8
  return %tr2 : i8
}

// CHECK-LABEL: @truncFPConstant
//       CHECK:   %[[cres:.+]] = arith.constant 1.000000e+00 : bf16
//       CHECK:   return %[[cres]]
func.func @truncFPConstant() -> bf16 {
  %cst = arith.constant 1.000000e+00 : f32
  %0 = arith.truncf %cst : f32 to bf16
  return %0 : bf16
}

// Test that cases with rounding are NOT propagated
// CHECK-LABEL: @truncFPConstantRounding
//       CHECK:   arith.constant 1.444000e+25 : f32
//       CHECK:   truncf
func.func @truncFPConstantRounding() -> bf16 {
  %cst = arith.constant 1.444000e+25 : f32
  %0 = arith.truncf %cst : f32 to bf16
  return %0 : bf16
}

// CHECK-LABEL: @tripleAddAdd
//       CHECK:   %[[cres:.+]] = arith.constant 59 : index
//       CHECK:   %[[add:.+]] = arith.addi %arg0, %[[cres]] : index
//       CHECK:   return %[[add]]
func.func @tripleAddAdd(%arg0: index) -> index {
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
func.func @tripleAddSub0(%arg0: index) -> index {
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
func.func @tripleAddSub1(%arg0: index) -> index {
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
func.func @tripleSubAdd0(%arg0: index) -> index {
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
func.func @tripleSubAdd1(%arg0: index) -> index {
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
func.func @tripleSubSub0(%arg0: index) -> index {
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
func.func @tripleSubSub1(%arg0: index) -> index {
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
func.func @tripleSubSub2(%arg0: index) -> index {
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
func.func @tripleSubSub3(%arg0: index) -> index {
  %c17 = arith.constant 17 : index
  %c42 = arith.constant 42 : index
  %add1 = arith.subi %arg0, %c17 : index
  %add2 = arith.subi %add1, %c42 : index
  return %add2 : index
}

// CHECK-LABEL: @doubleAddSub1
//  CHECK-NEXT:   return %arg0
func.func @doubleAddSub1(%arg0: index, %arg1 : index) -> index {
  %sub = arith.subi %arg0, %arg1 : index
  %add = arith.addi %sub, %arg1 : index
  return %add : index
}

// CHECK-LABEL: @doubleAddSub2
//  CHECK-NEXT:   return %arg0
func.func @doubleAddSub2(%arg0: index, %arg1 : index) -> index {
  %sub = arith.subi %arg0, %arg1 : index
  %add = arith.addi %arg1, %sub : index
  return %add : index
}

// CHECK-LABEL: @addiCarryZeroRhs
//  CHECK-NEXT:   %[[false:.+]] = arith.constant false
//  CHECK-NEXT:   return %arg0, %[[false]]
func.func @addiCarryZeroRhs(%arg0: i32) -> (i32, i1) {
  %zero = arith.constant 0 : i32
  %sum, %carry = arith.addi_carry %arg0, %zero: i32, i1
  return %sum, %carry : i32, i1
}

// CHECK-LABEL: @addiCarryZeroRhsSplat
//  CHECK-NEXT:   %[[false:.+]] = arith.constant dense<false> : vector<4xi1>
//  CHECK-NEXT:   return %arg0, %[[false]]
func.func @addiCarryZeroRhsSplat(%arg0: vector<4xi32>) -> (vector<4xi32>, vector<4xi1>) {
  %zero = arith.constant dense<0> : vector<4xi32>
  %sum, %carry = arith.addi_carry %arg0, %zero: vector<4xi32>, vector<4xi1>
  return %sum, %carry : vector<4xi32>, vector<4xi1>
}

// CHECK-LABEL: @addiCarryZeroLhs
//  CHECK-NEXT:   %[[false:.+]] = arith.constant false
//  CHECK-NEXT:   return %arg0, %[[false]]
func.func @addiCarryZeroLhs(%arg0: i32) -> (i32, i1) {
  %zero = arith.constant 0 : i32
  %sum, %carry = arith.addi_carry %zero, %arg0: i32, i1
  return %sum, %carry : i32, i1
}

// CHECK-LABEL: @addiCarryConstants
//  CHECK-DAG:    %[[false:.+]] = arith.constant false
//  CHECK-DAG:    %[[c50:.+]] = arith.constant 50 : i32
//  CHECK-NEXT:   return %[[c50]], %[[false]]
func.func @addiCarryConstants() -> (i32, i1) {
  %c13 = arith.constant 13 : i32
  %c37 = arith.constant 37 : i32
  %sum, %carry = arith.addi_carry %c13, %c37: i32, i1
  return %sum, %carry : i32, i1
}

// CHECK-LABEL: @addiCarryConstantsOverflow1
//  CHECK-DAG:    %[[true:.+]] = arith.constant true
//  CHECK-DAG:    %[[c0:.+]] = arith.constant 0 : i32
//  CHECK-NEXT:   return %[[c0]], %[[true]]
func.func @addiCarryConstantsOverflow1() -> (i32, i1) {
  %max = arith.constant 4294967295 : i32
  %c1 = arith.constant 1 : i32
  %sum, %carry = arith.addi_carry %max, %c1: i32, i1
  return %sum, %carry : i32, i1
}

// CHECK-LABEL: @addiCarryConstantsOverflow2
//  CHECK-DAG:    %[[true:.+]] = arith.constant true
//  CHECK-DAG:    %[[c_2:.+]] = arith.constant -2 : i32
// CHECK-NEXT:    return %[[c_2]], %[[true]]
func.func @addiCarryConstantsOverflow2() -> (i32, i1) {
  %max = arith.constant 4294967295 : i32
  %sum, %carry = arith.addi_carry %max, %max: i32, i1
  return %sum, %carry : i32, i1
}

// CHECK-LABEL: @addiCarryConstantsOverflowVector
//  CHECK-DAG:    %[[sum:.+]] = arith.constant dense<[1, 6, 2, 14]> : vector<4xi32>
//  CHECK-DAG:    %[[carry:.+]] = arith.constant dense<[false, false, true, false]> : vector<4xi1>
// CHECK-NEXT:    return %[[sum]], %[[carry]]
func.func @addiCarryConstantsOverflowVector() -> (vector<4xi32>, vector<4xi1>) {
  %v1 = arith.constant dense<[1, 3, 3, 7]> : vector<4xi32>
  %v2 = arith.constant dense<[0, 3, 4294967295, 7]> : vector<4xi32>
  %sum, %carry = arith.addi_carry %v1, %v2 : vector<4xi32>, vector<4xi1>
  return %sum, %carry : vector<4xi32>, vector<4xi1>
}

// CHECK-LABEL: @addiCarryConstantsSplatVector
//   CHECK-DAG:   %[[sum:.+]] = arith.constant dense<3> : vector<4xi32>
//   CHECK-DAG:   %[[carry:.+]] = arith.constant dense<false> : vector<4xi1>
//  CHECK-NEXT:   return %[[sum]], %[[carry]]
func.func @addiCarryConstantsSplatVector() -> (vector<4xi32>, vector<4xi1>) {
  %v1 = arith.constant dense<1> : vector<4xi32>
  %v2 = arith.constant dense<2> : vector<4xi32>
  %sum, %carry = arith.addi_carry %v1, %v2 : vector<4xi32>, vector<4xi1>
  return %sum, %carry : vector<4xi32>, vector<4xi1>
}

// CHECK-LABEL: @notCmpEQ
//       CHECK:   %[[cres:.+]] = arith.cmpi ne, %arg0, %arg1 : i8
//       CHECK:   return %[[cres]]
func.func @notCmpEQ(%arg0: i8, %arg1: i8) -> i1 {
  %true = arith.constant true
  %cmp = arith.cmpi "eq", %arg0, %arg1 : i8
  %ncmp = arith.xori %cmp, %true : i1
  return %ncmp : i1
}

// CHECK-LABEL: @notCmpEQ2
//       CHECK:   %[[cres:.+]] = arith.cmpi ne, %arg0, %arg1 : i8
//       CHECK:   return %[[cres]]
func.func @notCmpEQ2(%arg0: i8, %arg1: i8) -> i1 {
  %true = arith.constant true
  %cmp = arith.cmpi "eq", %arg0, %arg1 : i8
  %ncmp = arith.xori %true, %cmp : i1
  return %ncmp : i1
}

// CHECK-LABEL: @notCmpNE
//       CHECK:   %[[cres:.+]] = arith.cmpi eq, %arg0, %arg1 : i8
//       CHECK:   return %[[cres]]
func.func @notCmpNE(%arg0: i8, %arg1: i8) -> i1 {
  %true = arith.constant true
  %cmp = arith.cmpi "ne", %arg0, %arg1 : i8
  %ncmp = arith.xori %cmp, %true : i1
  return %ncmp : i1
}

// CHECK-LABEL: @notCmpSLT
//       CHECK:   %[[cres:.+]] = arith.cmpi sge, %arg0, %arg1 : i8
//       CHECK:   return %[[cres]]
func.func @notCmpSLT(%arg0: i8, %arg1: i8) -> i1 {
  %true = arith.constant true
  %cmp = arith.cmpi "slt", %arg0, %arg1 : i8
  %ncmp = arith.xori %cmp, %true : i1
  return %ncmp : i1
}

// CHECK-LABEL: @notCmpSLE
//       CHECK:   %[[cres:.+]] = arith.cmpi sgt, %arg0, %arg1 : i8
//       CHECK:   return %[[cres]]
func.func @notCmpSLE(%arg0: i8, %arg1: i8) -> i1 {
  %true = arith.constant true
  %cmp = arith.cmpi "sle", %arg0, %arg1 : i8
  %ncmp = arith.xori %cmp, %true : i1
  return %ncmp : i1
}

// CHECK-LABEL: @notCmpSGT
//       CHECK:   %[[cres:.+]] = arith.cmpi sle, %arg0, %arg1 : i8
//       CHECK:   return %[[cres]]
func.func @notCmpSGT(%arg0: i8, %arg1: i8) -> i1 {
  %true = arith.constant true
  %cmp = arith.cmpi "sgt", %arg0, %arg1 : i8
  %ncmp = arith.xori %cmp, %true : i1
  return %ncmp : i1
}

// CHECK-LABEL: @notCmpSGE
//       CHECK:   %[[cres:.+]] = arith.cmpi slt, %arg0, %arg1 : i8
//       CHECK:   return %[[cres]]
func.func @notCmpSGE(%arg0: i8, %arg1: i8) -> i1 {
  %true = arith.constant true
  %cmp = arith.cmpi "sge", %arg0, %arg1 : i8
  %ncmp = arith.xori %cmp, %true : i1
  return %ncmp : i1
}

// CHECK-LABEL: @notCmpULT
//       CHECK:   %[[cres:.+]] = arith.cmpi uge, %arg0, %arg1 : i8
//       CHECK:   return %[[cres]]
func.func @notCmpULT(%arg0: i8, %arg1: i8) -> i1 {
  %true = arith.constant true
  %cmp = arith.cmpi "ult", %arg0, %arg1 : i8
  %ncmp = arith.xori %cmp, %true : i1
  return %ncmp : i1
}

// CHECK-LABEL: @notCmpULE
//       CHECK:   %[[cres:.+]] = arith.cmpi ugt, %arg0, %arg1 : i8
//       CHECK:   return %[[cres]]
func.func @notCmpULE(%arg0: i8, %arg1: i8) -> i1 {
  %true = arith.constant true
  %cmp = arith.cmpi "ule", %arg0, %arg1 : i8
  %ncmp = arith.xori %cmp, %true : i1
  return %ncmp : i1
}

// CHECK-LABEL: @notCmpUGT
//       CHECK:   %[[cres:.+]] = arith.cmpi ule, %arg0, %arg1 : i8
//       CHECK:   return %[[cres]]
func.func @notCmpUGT(%arg0: i8, %arg1: i8) -> i1 {
  %true = arith.constant true
  %cmp = arith.cmpi "ugt", %arg0, %arg1 : i8
  %ncmp = arith.xori %cmp, %true : i1
  return %ncmp : i1
}

// CHECK-LABEL: @notCmpUGE
//       CHECK:   %[[cres:.+]] = arith.cmpi ult, %arg0, %arg1 : i8
//       CHECK:   return %[[cres]]
func.func @notCmpUGE(%arg0: i8, %arg1: i8) -> i1 {
  %true = arith.constant true
  %cmp = arith.cmpi "uge", %arg0, %arg1 : i8
  %ncmp = arith.xori %cmp, %true : i1
  return %ncmp : i1
}

// -----

// CHECK-LABEL: @xorxor(
//       CHECK-NOT: xori
//       CHECK:   return %arg0
func.func @xorxor(%cmp : i1) -> i1 {
  %true = arith.constant true
  %ncmp = arith.xori %cmp, %true : i1
  %nncmp = arith.xori %ncmp, %true : i1
  return %nncmp : i1
}

// -----

// CHECK-LABEL: @bitcastSameType(
// CHECK-SAME: %[[ARG:[a-zA-Z0-9_]*]]
func.func @bitcastSameType(%arg : f32) -> f32 {
  // CHECK: return %[[ARG]]
  %res = arith.bitcast %arg : f32 to f32
  return %res : f32
}

// -----

// CHECK-LABEL: @bitcastConstantFPtoI(
func.func @bitcastConstantFPtoI() -> i32 {
  // CHECK: %[[C0:.+]] = arith.constant 0 : i32
  // CHECK: return %[[C0]]
  %c0 = arith.constant 0.0 : f32
  %res = arith.bitcast %c0 : f32 to i32
  return %res : i32
}

// -----

// CHECK-LABEL: @bitcastConstantItoFP(
func.func @bitcastConstantItoFP() -> f32 {
  // CHECK: %[[C0:.+]] = arith.constant 0.0{{.*}} : f32
  // CHECK: return %[[C0]]
  %c0 = arith.constant 0 : i32
  %res = arith.bitcast %c0 : i32 to f32
  return %res : f32
}

// -----

// CHECK-LABEL: @bitcastConstantFPtoFP(
func.func @bitcastConstantFPtoFP() -> f16 {
  // CHECK: %[[C0:.+]] = arith.constant 0.0{{.*}} : f16
  // CHECK: return %[[C0]]
  %c0 = arith.constant 0.0 : bf16
  %res = arith.bitcast %c0 : bf16 to f16
  return %res : f16
}

// -----

// CHECK-LABEL: @bitcastConstantVecFPtoI(
func.func @bitcastConstantVecFPtoI() -> vector<3xf32> {
  // CHECK: %[[C0:.+]] = arith.constant dense<0.0{{.*}}> : vector<3xf32>
  // CHECK: return %[[C0]]
  %c0 = arith.constant dense<0> : vector<3xi32>
  %res = arith.bitcast %c0 : vector<3xi32> to vector<3xf32>
  return %res : vector<3xf32>
}

// -----

// CHECK-LABEL: @bitcastConstantVecItoFP(
func.func @bitcastConstantVecItoFP() -> vector<3xi32> {
  // CHECK: %[[C0:.+]] = arith.constant dense<0> : vector<3xi32>
  // CHECK: return %[[C0]]
  %c0 = arith.constant dense<0.0> : vector<3xf32>
  %res = arith.bitcast %c0 : vector<3xf32> to vector<3xi32>
  return %res : vector<3xi32>
}

// -----

// CHECK-LABEL: @bitcastConstantVecFPtoFP(
func.func @bitcastConstantVecFPtoFP() -> vector<3xbf16> {
  // CHECK: %[[C0:.+]] = arith.constant dense<0.0{{.*}}> : vector<3xbf16>
  // CHECK: return %[[C0]]
  %c0 = arith.constant dense<0.0> : vector<3xf16>
  %res = arith.bitcast %c0 : vector<3xf16> to vector<3xbf16>
  return %res : vector<3xbf16>
}

// -----

// CHECK-LABEL: @bitcastBackAndForth(
// CHECK-SAME: %[[ARG:[a-zA-Z0-9_]*]]
func.func @bitcastBackAndForth(%arg : i32) -> i32 {
  // CHECK: return %[[ARG]]
  %f = arith.bitcast %arg : i32 to f32
  %res = arith.bitcast %f : f32 to i32
  return %res : i32
}

// -----

// CHECK-LABEL: @bitcastOfBitcast(
// CHECK-SAME: %[[ARG:[a-zA-Z0-9_]*]]
func.func @bitcastOfBitcast(%arg : i16) -> i16 {
  // CHECK: return %[[ARG]]
  %f = arith.bitcast %arg : i16 to f16
  %bf = arith.bitcast %f : f16 to bf16
  %res = arith.bitcast %bf : bf16 to i16
  return %res : i16
}

// -----

// CHECK-LABEL: test_maxsi
// CHECK-DAG: %[[C0:.+]] = arith.constant 42
// CHECK-DAG: %[[MAX_INT_CST:.+]] = arith.constant 127
// CHECK: %[[X:.+]] = arith.maxsi %arg0, %[[C0]]
// CHECK: return %arg0, %[[MAX_INT_CST]], %arg0, %[[X]]
func.func @test_maxsi(%arg0 : i8) -> (i8, i8, i8, i8) {
  %maxIntCst = arith.constant 127 : i8
  %minIntCst = arith.constant -128 : i8
  %c0 = arith.constant 42 : i8
  %0 = arith.maxsi %arg0, %arg0 : i8
  %1 = arith.maxsi %arg0, %maxIntCst : i8
  %2 = arith.maxsi %arg0, %minIntCst : i8
  %3 = arith.maxsi %arg0, %c0 : i8
  return %0, %1, %2, %3: i8, i8, i8, i8
}

// CHECK-LABEL: test_maxsi2
// CHECK-DAG: %[[C0:.+]] = arith.constant 42
// CHECK-DAG: %[[MAX_INT_CST:.+]] = arith.constant 127
// CHECK: %[[X:.+]] = arith.maxsi %arg0, %[[C0]]
// CHECK: return %arg0, %[[MAX_INT_CST]], %arg0, %[[X]]
func.func @test_maxsi2(%arg0 : i8) -> (i8, i8, i8, i8) {
  %maxIntCst = arith.constant 127 : i8
  %minIntCst = arith.constant -128 : i8
  %c0 = arith.constant 42 : i8
  %0 = arith.maxsi %arg0, %arg0 : i8
  %1 = arith.maxsi %maxIntCst, %arg0: i8
  %2 = arith.maxsi %minIntCst, %arg0: i8
  %3 = arith.maxsi %c0, %arg0 : i8
  return %0, %1, %2, %3: i8, i8, i8, i8
}

// -----

// CHECK-LABEL: test_maxui
// CHECK-DAG: %[[C0:.+]] = arith.constant 42
// CHECK-DAG: %[[MAX_INT_CST:.+]] = arith.constant -1
// CHECK: %[[X:.+]] = arith.maxui %arg0, %[[C0]]
// CHECK: return %arg0, %[[MAX_INT_CST]], %arg0, %[[X]]
func.func @test_maxui(%arg0 : i8) -> (i8, i8, i8, i8) {
  %maxIntCst = arith.constant 255 : i8
  %minIntCst = arith.constant 0 : i8
  %c0 = arith.constant 42 : i8
  %0 = arith.maxui %arg0, %arg0 : i8
  %1 = arith.maxui %arg0, %maxIntCst : i8
  %2 = arith.maxui %arg0, %minIntCst : i8
  %3 = arith.maxui %arg0, %c0 : i8
  return %0, %1, %2, %3: i8, i8, i8, i8
}

// CHECK-LABEL: test_maxui
// CHECK-DAG: %[[C0:.+]] = arith.constant 42
// CHECK-DAG: %[[MAX_INT_CST:.+]] = arith.constant -1
// CHECK: %[[X:.+]] = arith.maxui %arg0, %[[C0]]
// CHECK: return %arg0, %[[MAX_INT_CST]], %arg0, %[[X]]
func.func @test_maxui2(%arg0 : i8) -> (i8, i8, i8, i8) {
  %maxIntCst = arith.constant 255 : i8
  %minIntCst = arith.constant 0 : i8
  %c0 = arith.constant 42 : i8
  %0 = arith.maxui %arg0, %arg0 : i8
  %1 = arith.maxui %maxIntCst, %arg0 : i8
  %2 = arith.maxui %minIntCst, %arg0 : i8
  %3 = arith.maxui %c0, %arg0 : i8
  return %0, %1, %2, %3: i8, i8, i8, i8
}

// -----

// CHECK-LABEL: test_minsi
// CHECK-DAG: %[[C0:.+]] = arith.constant 42
// CHECK-DAG: %[[MIN_INT_CST:.+]] = arith.constant -128
// CHECK: %[[X:.+]] = arith.minsi %arg0, %[[C0]]
// CHECK: return %arg0, %arg0, %[[MIN_INT_CST]], %[[X]]
func.func @test_minsi(%arg0 : i8) -> (i8, i8, i8, i8) {
  %maxIntCst = arith.constant 127 : i8
  %minIntCst = arith.constant -128 : i8
  %c0 = arith.constant 42 : i8
  %0 = arith.minsi %arg0, %arg0 : i8
  %1 = arith.minsi %arg0, %maxIntCst : i8
  %2 = arith.minsi %arg0, %minIntCst : i8
  %3 = arith.minsi %arg0, %c0 : i8
  return %0, %1, %2, %3: i8, i8, i8, i8
}

// CHECK-LABEL: test_minsi
// CHECK-DAG: %[[C0:.+]] = arith.constant 42
// CHECK-DAG: %[[MIN_INT_CST:.+]] = arith.constant -128
// CHECK: %[[X:.+]] = arith.minsi %arg0, %[[C0]]
// CHECK: return %arg0, %arg0, %[[MIN_INT_CST]], %[[X]]
func.func @test_minsi2(%arg0 : i8) -> (i8, i8, i8, i8) {
  %maxIntCst = arith.constant 127 : i8
  %minIntCst = arith.constant -128 : i8
  %c0 = arith.constant 42 : i8
  %0 = arith.minsi %arg0, %arg0 : i8
  %1 = arith.minsi %maxIntCst, %arg0 : i8
  %2 = arith.minsi %minIntCst, %arg0 : i8
  %3 = arith.minsi %c0, %arg0 : i8
  return %0, %1, %2, %3: i8, i8, i8, i8
}

// -----

// CHECK-LABEL: test_minui
// CHECK-DAG: %[[C0:.+]] = arith.constant 42
// CHECK-DAG: %[[MIN_INT_CST:.+]] = arith.constant 0
// CHECK: %[[X:.+]] = arith.minui %arg0, %[[C0]]
// CHECK: return %arg0, %arg0, %[[MIN_INT_CST]], %[[X]]
func.func @test_minui(%arg0 : i8) -> (i8, i8, i8, i8) {
  %maxIntCst = arith.constant 255 : i8
  %minIntCst = arith.constant 0 : i8
  %c0 = arith.constant 42 : i8
  %0 = arith.minui %arg0, %arg0 : i8
  %1 = arith.minui %arg0, %maxIntCst : i8
  %2 = arith.minui %arg0, %minIntCst : i8
  %3 = arith.minui %arg0, %c0 : i8
  return %0, %1, %2, %3: i8, i8, i8, i8
}

// CHECK-LABEL: test_minui
// CHECK-DAG: %[[C0:.+]] = arith.constant 42
// CHECK-DAG: %[[MIN_INT_CST:.+]] = arith.constant 0
// CHECK: %[[X:.+]] = arith.minui %arg0, %[[C0]]
// CHECK: return %arg0, %arg0, %[[MIN_INT_CST]], %[[X]]
func.func @test_minui2(%arg0 : i8) -> (i8, i8, i8, i8) {
  %maxIntCst = arith.constant 255 : i8
  %minIntCst = arith.constant 0 : i8
  %c0 = arith.constant 42 : i8
  %0 = arith.minui %arg0, %arg0 : i8
  %1 = arith.minui %maxIntCst, %arg0 : i8
  %2 = arith.minui %minIntCst, %arg0 : i8
  %3 = arith.minui %c0, %arg0 : i8
  return %0, %1, %2, %3: i8, i8, i8, i8
}

// -----

// CHECK-LABEL: @test_minf(
func.func @test_minf(%arg0 : f32) -> (f32, f32, f32) {
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
func.func @test_maxf(%arg0 : f32) -> (f32, f32, f32) {
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
func.func @test_addf(%arg0 : f32) -> (f32, f32, f32, f32) {
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
func.func @test_subf(%arg0 : f16) -> (f16, f16, f16) {
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
func.func @test_mulf(%arg0 : f32) -> (f32, f32, f32, f32) {
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

// CHECK-LABEL: @test_mulf1(
func.func @test_mulf1(%arg0 : f32, %arg1 : f32) -> (f32) {
  // CHECK-NEXT:  %[[X:.+]] = arith.mulf %arg0, %arg1 : f32
  // CHECK-NEXT:  return %[[X]]
  %0 = arith.negf %arg0 : f32
  %1 = arith.negf %arg1 : f32
  %2 = arith.mulf %0, %1 : f32
  return %2 : f32
}

// -----

// CHECK-LABEL: @test_divf(
func.func @test_divf(%arg0 : f64) -> (f64, f64) {
  // CHECK-NEXT:  %[[C5:.+]] = arith.constant 5.000000e-01
  // CHECK-NEXT:   return %arg0, %[[C5]]
  %c1 = arith.constant 1.0 : f64
  %c2 = arith.constant 2.0 : f64
  %0 = arith.divf %arg0, %c1 : f64
  %1 = arith.divf %c1, %c2 : f64
  return %0, %1 : f64, f64
}

// CHECK-LABEL: @test_divf1(
func.func @test_divf1(%arg0 : f32, %arg1 : f32) -> (f32) {
  // CHECK-NEXT:  %[[X:.+]] = arith.divf %arg0, %arg1 : f32
  // CHECK-NEXT:  return %[[X]]
  %0 = arith.negf %arg0 : f32
  %1 = arith.negf %arg1 : f32
  %2 = arith.divf %0, %1 : f32
  return %2 : f32
}

// -----

// CHECK-LABEL: @test_cmpf(
func.func @test_cmpf(%arg0 : f32) -> (i1, i1, i1, i1) {
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
func.func @constant_FPtoUI() -> i32 {
  // CHECK: %[[C0:.+]] = arith.constant 2 : i32
  // CHECK: return %[[C0]]
  %c0 = arith.constant 2.0 : f32
  %res = arith.fptoui %c0 : f32 to i32
  return %res : i32
}

// CHECK-LABEL: @constant_FPtoUI_splat(
func.func @constant_FPtoUI_splat() -> vector<4xi32> {
  // CHECK: %[[C0:.+]] = arith.constant dense<2> : vector<4xi32>
  // CHECK: return %[[C0]]
  %c0 = arith.constant 2.0 : f32
  %splat = vector.splat %c0 : vector<4xf32>
  %res = arith.fptoui %splat : vector<4xf32> to vector<4xi32>
  return %res : vector<4xi32>
}

// CHECK-LABEL: @constant_FPtoUI_vector(
func.func @constant_FPtoUI_vector() -> vector<4xi32> {
  // CHECK: %[[C0:.+]] = arith.constant dense<[1, 3, 5, 7]> : vector<4xi32>
  // CHECK: return %[[C0]]
  %vector = arith.constant dense<[1.0, 3.0, 5.0, 7.0]> : vector<4xf32>
  %res = arith.fptoui %vector : vector<4xf32> to vector<4xi32>
  return %res : vector<4xi32>
}

// -----
// CHECK-LABEL: @invalid_constant_FPtoUI(
func.func @invalid_constant_FPtoUI() -> i32 {
  // CHECK: %[[C0:.+]] = arith.constant -2.000000e+00 : f32
  // CHECK: %[[C1:.+]] = arith.fptoui %[[C0]] : f32 to i32
  // CHECK: return %[[C1]]
  %c0 = arith.constant -2.0 : f32
  %res = arith.fptoui %c0 : f32 to i32
  return %res : i32
}

// -----
// CHECK-LABEL: @constant_FPtoSI(
func.func @constant_FPtoSI() -> i32 {
  // CHECK: %[[C0:.+]] = arith.constant -2 : i32
  // CHECK: return %[[C0]]
  %c0 = arith.constant -2.0 : f32
  %res = arith.fptosi %c0 : f32 to i32
  return %res : i32
}

// CHECK-LABEL: @constant_FPtoSI_splat(
func.func @constant_FPtoSI_splat() -> vector<4xi32> {
  // CHECK: %[[C0:.+]] = arith.constant dense<-2> : vector<4xi32>
  // CHECK: return %[[C0]]
  %c0 = arith.constant -2.0 : f32
  %splat = vector.splat %c0 : vector<4xf32>
  %res = arith.fptosi %splat : vector<4xf32> to vector<4xi32>
  return %res : vector<4xi32>
}

// CHECK-LABEL: @constant_FPtoSI_vector(
func.func @constant_FPtoSI_vector() -> vector<4xi32> {
  // CHECK: %[[C0:.+]] = arith.constant dense<[-1, -3, -5, -7]> : vector<4xi32>
  // CHECK: return %[[C0]]
  %vector = arith.constant dense<[-1.0, -3.0, -5.0, -7.0]> : vector<4xf32>
  %res = arith.fptosi %vector : vector<4xf32> to vector<4xi32>
  return %res : vector<4xi32>
}

// -----
// CHECK-LABEL: @invalid_constant_FPtoSI(
func.func @invalid_constant_FPtoSI() -> i8 {
  // CHECK: %[[C0:.+]] = arith.constant 2.000000e+10 : f32
  // CHECK: %[[C1:.+]] = arith.fptosi %[[C0]] : f32 to i8
  // CHECK: return %[[C1]]
  %c0 = arith.constant 2.0e10 : f32
  %res = arith.fptosi %c0 : f32 to i8
  return %res : i8
}

// CHECK-LABEL: @constant_SItoFP(
func.func @constant_SItoFP() -> f32 {
  // CHECK: %[[C0:.+]] = arith.constant -2.000000e+00 : f32
  // CHECK: return %[[C0]]
  %c0 = arith.constant -2 : i32
  %res = arith.sitofp %c0 : i32 to f32
  return %res : f32
}

// CHECK-LABEL: @constant_SItoFP_splat(
func.func @constant_SItoFP_splat() -> vector<4xf32> {
  // CHECK: %[[C0:.+]] = arith.constant dense<2.000000e+00> : vector<4xf32>
  // CHECK: return %[[C0]]
  %c0 = arith.constant 2 : i32
  %splat = vector.splat %c0 : vector<4xi32>
  %res = arith.sitofp %splat : vector<4xi32> to vector<4xf32>
  return %res : vector<4xf32>
}

// CHECK-LABEL: @constant_SItoFP_vector(
func.func @constant_SItoFP_vector() -> vector<4xf32> {
  // CHECK: %[[C0:.+]] = arith.constant dense<[1.000000e+00, 3.000000e+00, 5.000000e+00, 7.000000e+00]> : vector<4xf32>
  // CHECK: return %[[C0]]
  %vector = arith.constant dense<[1, 3, 5, 7]> : vector<4xi32>
  %res = arith.sitofp %vector : vector<4xi32> to vector<4xf32>
  return %res : vector<4xf32>
}

// -----
// CHECK-LABEL: @constant_UItoFP(
func.func @constant_UItoFP() -> f32 {
  // CHECK: %[[C0:.+]] = arith.constant 2.000000e+00 : f32
  // CHECK: return %[[C0]]
  %c0 = arith.constant 2 : i32
  %res = arith.uitofp %c0 : i32 to f32
  return %res : f32
}

// CHECK-LABEL: @constant_UItoFP_splat(
func.func @constant_UItoFP_splat() -> vector<4xf32> {
  // CHECK: %[[C0:.+]] = arith.constant dense<2.000000e+00> : vector<4xf32>
  // CHECK: return %[[C0]]
  %c0 = arith.constant 2 : i32
  %splat = vector.splat %c0 : vector<4xi32>
  %res = arith.uitofp %splat : vector<4xi32> to vector<4xf32>
  return %res : vector<4xf32>
}

// CHECK-LABEL: @constant_UItoFP_vector(
func.func @constant_UItoFP_vector() -> vector<4xf32> {
  // CHECK: %[[C0:.+]] = arith.constant dense<[1.000000e+00, 3.000000e+00, 5.000000e+00, 7.000000e+00]> : vector<4xf32>
  // CHECK: return %[[C0]]
  %vector = arith.constant dense<[1, 3, 5, 7]> : vector<4xi32>
  %res = arith.uitofp %vector : vector<4xi32> to vector<4xf32>
  return %res : vector<4xf32>
}

// -----

// Tests rewritten from https://github.com/llvm/llvm-project/blob/main/llvm/test/Transforms/InstCombine/2008-11-08-FCmp.ll
// When inst combining an FCMP with the LHS coming from a arith.uitofp instruction, we
// can lower it to signed ICMP instructions.

// CHECK-LABEL: @test1(
// CHECK-SAME: %[[arg0:.+]]:
func.func @test1(%arg0: i32) -> i1 {
  %cst = arith.constant 0.000000e+00 : f64
  %1 = arith.uitofp %arg0: i32 to f64
  %2 = arith.cmpf ole, %1, %cst : f64
  // CHECK: %[[c0:.+]] = arith.constant 0 : i32
  // CHECK: arith.cmpi ule, %[[arg0]], %[[c0]] : i32
  return %2 : i1
}

// CHECK-LABEL: @test2(
// CHECK-SAME: %[[arg0:.+]]:
func.func @test2(%arg0: i32) -> i1 {
  %cst = arith.constant 0.000000e+00 : f64
  %1 = arith.uitofp %arg0: i32 to f64
  %2 = arith.cmpf olt, %1, %cst : f64
  return %2 : i1
  // CHECK: %[[c0:.+]] = arith.constant 0 : i32
  // CHECK: arith.cmpi ult, %[[arg0]], %[[c0]] : i32
}

// CHECK-LABEL: @test3(
// CHECK-SAME: %[[arg0:.+]]:
func.func @test3(%arg0: i32) -> i1 {
  %cst = arith.constant 0.000000e+00 : f64
  %1 = arith.uitofp %arg0: i32 to f64
  %2 = arith.cmpf oge, %1, %cst : f64
  return %2 : i1
  // CHECK: %[[c0:.+]] = arith.constant 0 : i32
  // CHECK: arith.cmpi uge, %[[arg0]], %[[c0]] : i32
}

// CHECK-LABEL: @test4(
// CHECK-SAME: %[[arg0:.+]]:
func.func @test4(%arg0: i32) -> i1 {
  %cst = arith.constant 0.000000e+00 : f64
  %1 = arith.uitofp %arg0: i32 to f64
  %2 = arith.cmpf ogt, %1, %cst : f64
  // CHECK: %[[c0:.+]] = arith.constant 0 : i32
  // CHECK: arith.cmpi ugt, %[[arg0]], %[[c0]] : i32
  return %2 : i1
}

// CHECK-LABEL: @test5(
func.func @test5(%arg0: i32) -> i1 {
  %cst = arith.constant -4.400000e+00 : f64
  %1 = arith.uitofp %arg0: i32 to f64
  %2 = arith.cmpf ogt, %1, %cst : f64
  return %2 : i1
  // CHECK: %[[true:.+]] = arith.constant true
  // CHECK: return %[[true]] : i1
}

// CHECK-LABEL: @test6(
func.func @test6(%arg0: i32) -> i1 {
  %cst = arith.constant -4.400000e+00 : f64
  %1 = arith.uitofp %arg0: i32 to f64
  %2 = arith.cmpf olt, %1, %cst : f64
  return %2 : i1
  // CHECK: %[[false:.+]] = arith.constant false
  // CHECK: return %[[false]] : i1
}

// Check that optimizing unsigned >= comparisons correctly distinguishes
// positive and negative constants.
// CHECK-LABEL: @test7(
// CHECK-SAME: %[[arg0:.+]]:
func.func @test7(%arg0: i32) -> i1 {
  %cst = arith.constant 3.200000e+00 : f64
  %1 = arith.uitofp %arg0: i32 to f64
  %2 = arith.cmpf oge, %1, %cst : f64
  return %2 : i1
  // CHECK: %[[c3:.+]] = arith.constant 3 : i32
  // CHECK: arith.cmpi ugt, %[[arg0]], %[[c3]] : i32
}

// -----

// CHECK-LABEL: @foldShl(
// CHECK: %[[res:.+]] = arith.constant 4294967296 : i64
// CHECK: return %[[res]]
func.func @foldShl() -> i64 {
  %c1 = arith.constant 1 : i64
  %c32 = arith.constant 32 : i64
  %r = arith.shli %c1, %c32 : i64
  return %r : i64
}

// CHECK-LABEL: @nofoldShl(
// CHECK: %[[res:.+]] = arith.shli
// CHECK: return %[[res]]
func.func @nofoldShl() -> i64 {
  %c1 = arith.constant 1 : i64
  %c132 = arith.constant 132 : i64
  %r = arith.shli %c1, %c132 : i64
  return %r : i64
}

// CHECK-LABEL: @nofoldShl2(
// CHECK: %[[res:.+]] = arith.shli
// CHECK: return %[[res]]
func.func @nofoldShl2() -> i64 {
  %c1 = arith.constant 1 : i64
  %cm32 = arith.constant -32 : i64
  %r = arith.shli %c1, %cm32 : i64
  return %r : i64
}

// CHECK-LABEL: @foldShru(
// CHECK: %[[res:.+]] = arith.constant 2 : i64
// CHECK: return %[[res]]
func.func @foldShru() -> i64 {
  %c1 = arith.constant 8 : i64
  %c32 = arith.constant 2 : i64
  %r = arith.shrui %c1, %c32 : i64
  return %r : i64
}

// CHECK-LABEL: @foldShru2(
// CHECK: %[[res:.+]] = arith.constant 9223372036854775807 : i64
// CHECK: return %[[res]]
func.func @foldShru2() -> i64 {
  %c1 = arith.constant -2 : i64
  %c32 = arith.constant 1 : i64
  %r = arith.shrui %c1, %c32 : i64
  return %r : i64
}

// CHECK-LABEL: @nofoldShru(
// CHECK: %[[res:.+]] = arith.shrui
// CHECK: return %[[res]]
func.func @nofoldShru() -> i64 {
  %c1 = arith.constant 8 : i64
  %c132 = arith.constant 132 : i64
  %r = arith.shrui %c1, %c132 : i64
  return %r : i64
}

// CHECK-LABEL: @nofoldShru2(
// CHECK: %[[res:.+]] = arith.shrui
// CHECK: return %[[res]]
func.func @nofoldShru2() -> i64 {
  %c1 = arith.constant 8 : i64
  %cm32 = arith.constant -32 : i64
  %r = arith.shrui %c1, %cm32 : i64
  return %r : i64
}

// CHECK-LABEL: @foldShrs(
// CHECK: %[[res:.+]] = arith.constant 2 : i64
// CHECK: return %[[res]]
func.func @foldShrs() -> i64 {
  %c1 = arith.constant 8 : i64
  %c32 = arith.constant 2 : i64
  %r = arith.shrsi %c1, %c32 : i64
  return %r : i64
}

// CHECK-LABEL: @foldShrs2(
// CHECK: %[[res:.+]] = arith.constant -1 : i64
// CHECK: return %[[res]]
func.func @foldShrs2() -> i64 {
  %c1 = arith.constant -2 : i64
  %c32 = arith.constant 1 : i64
  %r = arith.shrsi %c1, %c32 : i64
  return %r : i64
}

// CHECK-LABEL: @nofoldShrs(
// CHECK: %[[res:.+]] = arith.shrsi
// CHECK: return %[[res]]
func.func @nofoldShrs() -> i64 {
  %c1 = arith.constant 8 : i64
  %c132 = arith.constant 132 : i64
  %r = arith.shrsi %c1, %c132 : i64
  return %r : i64
}

// CHECK-LABEL: @nofoldShrs2(
// CHECK: %[[res:.+]] = arith.shrsi
// CHECK: return %[[res]]
func.func @nofoldShrs2() -> i64 {
  %c1 = arith.constant 8 : i64
  %cm32 = arith.constant -32 : i64
  %r = arith.shrsi %c1, %cm32 : i64
  return %r : i64
}

// -----

// CHECK-LABEL: @test_negf(
// CHECK: %[[res:.+]] = arith.constant -2.0
// CHECK: return %[[res]]
func.func @test_negf() -> (f32) {
  %c = arith.constant 2.0 : f32
  %0 = arith.negf %c : f32
  return %0: f32
}

// CHECK-LABEL: @test_negf1(
// CHECK-SAME: %[[arg0:.+]]:
// CHECK: return %[[arg0]]
func.func @test_negf1(%f : f32) -> (f32) {
  %0 = arith.negf %f : f32
  %1 = arith.negf %0 : f32
  return %1: f32
}

// -----

// CHECK-LABEL: @test_remui(
// CHECK: %[[res:.+]] = arith.constant dense<[0, 0, 4, 2]> : vector<4xi32>
// CHECK: return %[[res]]
func.func @test_remui() -> (vector<4xi32>) {
  %v1 = arith.constant dense<[9, 9, 9, 9]> : vector<4xi32>
  %v2 = arith.constant dense<[1, 3, 5, 7]> : vector<4xi32>
  %0 = arith.remui %v1, %v2 : vector<4xi32>
  return %0 : vector<4xi32>
}

// // -----

// CHECK-LABEL: @test_remui_1(
// CHECK: %[[res:.+]] = arith.constant dense<0> : vector<4xi32>
// CHECK: return %[[res]]
func.func @test_remui_1(%arg : vector<4xi32>) -> (vector<4xi32>) {
  %v = arith.constant dense<[1, 1, 1, 1]> : vector<4xi32>
  %0 = arith.remui %arg, %v : vector<4xi32>
  return %0 : vector<4xi32>
}

// -----

// CHECK-LABEL: @test_remsi(
// CHECK: %[[res:.+]] = arith.constant dense<[0, 0, 4, 2]> : vector<4xi32>
// CHECK: return %[[res]]
func.func @test_remsi() -> (vector<4xi32>) {
  %v1 = arith.constant dense<[9, 9, 9, 9]> : vector<4xi32>
  %v2 = arith.constant dense<[1, 3, 5, 7]> : vector<4xi32>
  %0 = arith.remsi %v1, %v2 : vector<4xi32>
  return %0 : vector<4xi32>
}

// // -----

// CHECK-LABEL: @test_remsi_1(
// CHECK: %[[res:.+]] = arith.constant dense<0> : vector<4xi32>
// CHECK: return %[[res]]
func.func @test_remsi_1(%arg : vector<4xi32>) -> (vector<4xi32>) {
  %v = arith.constant dense<[1, 1, 1, 1]> : vector<4xi32>
  %0 = arith.remsi %arg, %v : vector<4xi32>
  return %0 : vector<4xi32>
}

// -----

// CHECK-LABEL: @test_remf(
// CHECK: %[[res:.+]] = arith.constant -1.000000e+00 : f32
// CHECK: return %[[res]]
func.func @test_remf() -> (f32) {
  %v1 = arith.constant 3.0 : f32
  %v2 = arith.constant 2.0 : f32
  %0 = arith.remf %v1, %v2 : f32
  return %0 : f32
}

// CHECK-LABEL: @test_remf_vec(
// CHECK: %[[res:.+]] = arith.constant dense<[1.000000e+00, 0.000000e+00, -1.000000e+00, 0.000000e+00]> : vector<4xf32>
// CHECK: return %[[res]]
func.func @test_remf_vec() -> (vector<4xf32>) {
  %v1 = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : vector<4xf32>
  %v2 = arith.constant dense<[2.0, 2.0, 2.0, 2.0]> : vector<4xf32>
  %0 = arith.remf %v1, %v2 : vector<4xf32>
  return %0 : vector<4xf32>
}

// -----

// CHECK-LABEL: @test_andi_not_fold_rhs(
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: %[[C:.*]] = arith.constant 0 : index
// CHECK: return %[[C]]

func.func @test_andi_not_fold_rhs(%arg0 : index) -> index {
    %0 = arith.constant -1 : index
    %1 = arith.xori %arg0, %0 : index
    %2 = arith.andi %arg0, %1 : index
    return %2 : index
}


// CHECK-LABEL: @test_andi_not_fold_lhs(
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: %[[C:.*]] = arith.constant 0 : index
// CHECK: return %[[C]]

func.func @test_andi_not_fold_lhs(%arg0 : index) -> index {
    %0 = arith.constant -1 : index
    %1 = arith.xori %arg0, %0 : index
    %2 = arith.andi %1, %arg0 : index
    return %2 : index
}
