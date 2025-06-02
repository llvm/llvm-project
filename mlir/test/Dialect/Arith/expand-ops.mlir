// RUN: mlir-opt %s -arith-expand="include-bf16=true include-f8e8m0=true" -split-input-file | FileCheck %s

// Test ceil divide with signed integer
// CHECK-LABEL:       func @ceildivi
// CHECK-SAME:     ([[ARG0:%.+]]: i32, [[ARG1:%.+]]: i32) -> i32 {
func.func @ceildivi(%arg0: i32, %arg1: i32) -> (i32) {
  %res = arith.ceildivsi %arg0, %arg1 : i32
  return %res : i32

// CHECK:           [[ZERO:%.+]] = arith.constant 0 : i32
// CHECK:           [[ONE:%.+]] = arith.constant 1 : i32
// CHECK:           [[DIV:%.+]] = arith.divsi %arg0, %arg1 : i32
// CHECK:           [[MUL:%.+]] = arith.muli [[DIV]], %arg1 : i32
// CHECK:           [[NEXACT:%.+]] = arith.cmpi ne, %arg0, [[MUL]] : i32
// CHECK:           [[NNEG:%.+]] = arith.cmpi slt, %arg0, [[ZERO]] : i32
// CHECK:           [[MNEG:%.+]] = arith.cmpi slt, %arg1, [[ZERO]] : i32
// CHECK:           [[SAMESIGN:%.+]] = arith.cmpi eq, [[NNEG]], [[MNEG]] : i1
// CHECK:           [[SHOULDROUND:%.+]] = arith.andi [[NEXACT]], [[SAMESIGN]] : i1
// CHECK:           [[CEIL:%.+]] = arith.addi [[DIV]], [[ONE]] : i32
// CHECK:           [[RES:%.+]] = arith.select [[SHOULDROUND]], [[CEIL]], [[DIV]] : i32
}

// -----

// Test ceil divide with index type
// CHECK-LABEL:       func @ceildivi_index
// CHECK-SAME:     ([[ARG0:%.+]]: index, [[ARG1:%.+]]: index) -> index {
func.func @ceildivi_index(%arg0: index, %arg1: index) -> (index) {
  %res = arith.ceildivsi %arg0, %arg1 : index
  return %res : index

// CHECK:           [[ZERO:%.+]] = arith.constant 0 : index
// CHECK:           [[ONE:%.+]] = arith.constant 1 : index
// CHECK:           [[DIV:%.+]] = arith.divsi %arg0, %arg1 : index
// CHECK:           [[MUL:%.+]] = arith.muli [[DIV]], %arg1 : index
// CHECK:           [[NEXACT:%.+]] = arith.cmpi ne, %arg0, [[MUL]] : index
// CHECK:           [[NNEG:%.+]] = arith.cmpi slt, %arg0, [[ZERO]] : index
// CHECK:           [[MNEG:%.+]] = arith.cmpi slt, %arg1, [[ZERO]] : index
// CHECK:           [[SAMESIGN:%.+]] = arith.cmpi eq, [[NNEG]], [[MNEG]] : i1
// CHECK:           [[SHOULDROUND:%.+]] = arith.andi [[NEXACT]], [[SAMESIGN]] : i1
// CHECK:           [[CEIL:%.+]] = arith.addi [[DIV]], [[ONE]] : index
// CHECK:           [[RES:%.+]] = arith.select [[SHOULDROUND]], [[CEIL]], [[DIV]] : index

}

// -----

// Test floor divide with signed integer
// CHECK-LABEL:       func @floordivi
// CHECK-SAME:     ([[ARG0:%.+]]: i32, [[ARG1:%.+]]: i32) -> i32 {
func.func @floordivi(%arg0: i32, %arg1: i32) -> (i32) {
  %res = arith.floordivsi %arg0, %arg1 : i32
  return %res : i32
// CHECK:   %[[QUOTIENT:.*]] = arith.divsi %arg0, %arg1 : i32
// CHECK:   %[[PRODUCT:.*]] = arith.muli %[[QUOTIENT]], %arg1 : i32
// CHECK:   %[[NOT_EQ_PRODUCT:.*]] = arith.cmpi ne, %arg0, %[[PRODUCT]] : i32
// CHECK-DAG:   %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK:   %[[NEG_DIVISOR:.*]] = arith.cmpi slt, %arg0, %[[ZERO]] : i32
// CHECK:   %[[NEG_DIVIDEND:.*]] = arith.cmpi slt, %arg1, %[[ZERO]] : i32
// CHECK:   %[[OPPOSITE_SIGN:.*]] = arith.cmpi ne, %[[NEG_DIVISOR]], %[[NEG_DIVIDEND]] : i1
// CHECK:   %[[CONDITION:.*]] = arith.andi %[[NOT_EQ_PRODUCT]], %[[OPPOSITE_SIGN]] : i1
// CHECK-DAG:   %[[NEG_ONE:.*]] = arith.constant -1 : i32
// CHECK:   %[[MINUS_ONE:.*]] = arith.addi %[[QUOTIENT]], %[[NEG_ONE]] : i32
// CHECK:   %[[RES:.*]] = arith.select %[[CONDITION]], %[[MINUS_ONE]], %[[QUOTIENT]] : i32
}

// -----

// Test floor divide with index type
// CHECK-LABEL:       func @floordivi_index
// CHECK-SAME:     ([[ARG0:%.+]]: index, [[ARG1:%.+]]: index) -> index {
func.func @floordivi_index(%arg0: index, %arg1: index) -> (index) {
  %res = arith.floordivsi %arg0, %arg1 : index
  return %res : index
// CHECK:   %[[QUOTIENT:.*]] = arith.divsi %arg0, %arg1 : index
// CHECK:   %[[PRODUCT:.*]] = arith.muli %[[QUOTIENT]], %arg1 : index
// CHECK:   %[[NOT_EQ_PRODUCT:.*]] = arith.cmpi ne, %arg0, %[[PRODUCT]] : index
// CHECK-DAG:   %[[ZERO:.*]] = arith.constant 0 : index
// CHECK:   %[[NEG_DIVISOR:.*]] = arith.cmpi slt, %arg0, %[[ZERO]] : index
// CHECK:   %[[NEG_DIVIDEND:.*]] = arith.cmpi slt, %arg1, %[[ZERO]] : index
// CHECK:   %[[OPPOSITE_SIGN:.*]] = arith.cmpi ne, %[[NEG_DIVISOR]], %[[NEG_DIVIDEND]] : i1
// CHECK:   %[[CONDITION:.*]] = arith.andi %[[NOT_EQ_PRODUCT]], %[[OPPOSITE_SIGN]] : i1
// CHECK:   %[[NEG_ONE:.*]] = arith.constant -1 : index
// CHECK-DAG:   %[[MINUS_ONE:.*]] = arith.addi %[[QUOTIENT]], %[[NEG_ONE]] : index
// CHECK:   %[[RES:.*]] = arith.select %[[CONDITION]], %[[MINUS_ONE]], %[[QUOTIENT]] : index
}

// -----

// Test floor divide with vector
// CHECK-LABEL:   func.func @floordivi_vec(
// CHECK-SAME:                             %[[VAL_0:.*]]: vector<4xi32>,
// CHECK-SAME:                             %[[VAL_1:.*]]: vector<4xi32>) -> vector<4xi32> {
func.func @floordivi_vec(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> (vector<4xi32>) {
  %res = arith.floordivsi %arg0, %arg1 : vector<4xi32>
  return %res : vector<4xi32>
// CHECK:   %[[QUOTIENT:.*]] = arith.divsi %arg0, %arg1 : vector<4xi32>
// CHECK:   %[[PRODUCT:.*]] = arith.muli %[[QUOTIENT]], %arg1 : vector<4xi32>
// CHECK:   %[[NOT_EQ_PRODUCT:.*]] = arith.cmpi ne, %arg0, %[[PRODUCT]] : vector<4xi32>
// CHECK-DAG:   %[[ZERO:.*]] = arith.constant dense<0> : vector<4xi32>
// CHECK:   %[[NEG_DIVISOR:.*]] = arith.cmpi slt, %arg0, %[[ZERO]] : vector<4xi32>
// CHECK:   %[[NEG_DIVIDEND:.*]] = arith.cmpi slt, %arg1, %[[ZERO]] : vector<4xi32>
// CHECK:   %[[OPPOSITE_SIGN:.*]] = arith.cmpi ne, %[[NEG_DIVISOR]], %[[NEG_DIVIDEND]] : vector<4xi1>
// CHECK:   %[[CONDITION:.*]] = arith.andi %[[NOT_EQ_PRODUCT]], %[[OPPOSITE_SIGN]] : vector<4xi1>
// CHECK-DAG:   %[[NEG_ONE:.*]] = arith.constant dense<-1> : vector<4xi32>
// CHECK:   %[[MINUS_ONE:.*]] = arith.addi %[[QUOTIENT]], %[[NEG_ONE]] : vector<4xi32>
// CHECK:   %[[RES:.*]] = arith.select %[[CONDITION]], %[[MINUS_ONE]], %[[QUOTIENT]] : vector<4xi1>, vector<4xi32>
}

// -----

// Test ceil divide with unsigned integer
// CHECK-LABEL:       func @ceildivui
// CHECK-SAME:     ([[ARG0:%.+]]: i32, [[ARG1:%.+]]: i32) -> i32 {
func.func @ceildivui(%arg0: i32, %arg1: i32) -> (i32) {
  %res = arith.ceildivui %arg0, %arg1 : i32
  return %res : i32
// CHECK:           [[ZERO:%.+]] = arith.constant 0 : i32
// CHECK:           [[ISZERO:%.+]] = arith.cmpi eq, %arg0, [[ZERO]] : i32
// CHECK:           [[ONE:%.+]] = arith.constant 1 : i32
// CHECK:           [[SUB:%.+]] = arith.subi %arg0, [[ONE]] : i32
// CHECK:           [[DIV:%.+]] = arith.divui [[SUB]], %arg1 : i32
// CHECK:           [[REM:%.+]] = arith.addi [[DIV]], [[ONE]] : i32
// CHECK:           [[RES:%.+]] = arith.select [[ISZERO]], [[ZERO]], [[REM]] : i32
}

// -----

// Test unsigned ceil divide with index
// CHECK-LABEL:       func @ceildivui_index
// CHECK-SAME:     ([[ARG0:%.+]]: index, [[ARG1:%.+]]: index) -> index {
func.func @ceildivui_index(%arg0: index, %arg1: index) -> (index) {
  %res = arith.ceildivui %arg0, %arg1 : index
  return %res : index
// CHECK:           [[ZERO:%.+]] = arith.constant 0 : index
// CHECK:           [[ISZERO:%.+]] = arith.cmpi eq, %arg0, [[ZERO]] : index
// CHECK:           [[ONE:%.+]] = arith.constant 1 : index
// CHECK:           [[SUB:%.+]] = arith.subi %arg0, [[ONE]] : index
// CHECK:           [[DIV:%.+]] = arith.divui [[SUB]], %arg1 : index
// CHECK:           [[REM:%.+]] = arith.addi [[DIV]], [[ONE]] : index
// CHECK:           [[RES:%.+]] = arith.select [[ISZERO]], [[ZERO]], [[REM]] : index
}

// -----

// CHECK-LABEL: func @maximumf
func.func @maximumf(%a: f32, %b: f32) -> f32 {
  %result = arith.maximumf %a, %b : f32
  return %result : f32
}
// CHECK-SAME: %[[LHS:.*]]: f32, %[[RHS:.*]]: f32)
// CHECK-NEXT: %[[CMP:.*]] = arith.cmpf ugt, %[[LHS]], %[[RHS]] : f32
// CHECK-NEXT: %[[SELECT:.*]] = arith.select %[[CMP]], %[[LHS]], %[[RHS]] : f32
// CHECK-NEXT: %[[IS_NAN:.*]] = arith.cmpf uno, %[[RHS]], %[[RHS]] : f32
// CHECK-NEXT: %[[RESULT:.*]] = arith.select %[[IS_NAN]], %[[RHS]], %[[SELECT]] : f32
// CHECK-NEXT: return %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @maximumf_vector
func.func @maximumf_vector(%a: vector<4xf16>, %b: vector<4xf16>) -> vector<4xf16> {
  %result = arith.maximumf %a, %b : vector<4xf16>
  return %result : vector<4xf16>
}
// CHECK-SAME: %[[LHS:.*]]: vector<4xf16>, %[[RHS:.*]]: vector<4xf16>)
// CHECK-NEXT: %[[CMP:.*]] = arith.cmpf ugt, %[[LHS]], %[[RHS]] : vector<4xf16>
// CHECK-NEXT: %[[SELECT:.*]] = arith.select %[[CMP]], %[[LHS]], %[[RHS]]
// CHECK-NEXT: %[[IS_NAN:.*]] = arith.cmpf uno, %[[RHS]], %[[RHS]] : vector<4xf16>
// CHECK-NEXT: %[[RESULT:.*]] = arith.select %[[IS_NAN]], %[[RHS]], %[[SELECT]]
// CHECK-NEXT: return %[[RESULT]] : vector<4xf16>

// -----

// CHECK-LABEL: func @maxnumf
func.func @maxnumf(%a: f32, %b: f32) -> f32 {
  %result = arith.maxnumf %a, %b : f32
  return %result : f32
}

// CHECK-SAME: %[[LHS:.*]]: f32, %[[RHS:.*]]: f32)
// CHECK-NEXT: %[[CMP:.*]] = arith.cmpf ugt, %[[LHS]], %[[RHS]] : f32
// CHECK-NEXT: %[[SELECT:.*]] = arith.select %[[CMP]], %[[LHS]], %[[RHS]] : f32
// CHECK-NEXT: %[[IS_NAN:.*]] = arith.cmpf uno, %[[LHS]], %[[LHS]] : f32
// CHECK-NEXT: %[[RESULT:.*]] = arith.select %[[IS_NAN]], %[[RHS]], %[[SELECT]] : f32
// CHECK-NEXT: return %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @minimumf
func.func @minimumf(%a: f32, %b: f32) -> f32 {
  %result = arith.minimumf %a, %b : f32
  return %result : f32
}

// CHECK-SAME: %[[LHS:.*]]: f32, %[[RHS:.*]]: f32)
// CHECK-NEXT: %[[CMP:.*]] = arith.cmpf ult, %[[LHS]], %[[RHS]] : f32
// CHECK-NEXT: %[[SELECT:.*]] = arith.select %[[CMP]], %[[LHS]], %[[RHS]] : f32
// CHECK-NEXT: %[[IS_NAN:.*]] = arith.cmpf uno, %[[RHS]], %[[RHS]] : f32
// CHECK-NEXT: %[[RESULT:.*]] = arith.select %[[IS_NAN]], %[[RHS]], %[[SELECT]] : f32
// CHECK-NEXT: return %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @minnumf
func.func @minnumf(%a: f32, %b: f32) -> f32 {
  %result = arith.minnumf %a, %b : f32
  return %result : f32
}

// CHECK-SAME: %[[LHS:.*]]: f32, %[[RHS:.*]]: f32)
// CHECK-NEXT: %[[CMP:.*]] = arith.cmpf ult, %[[LHS]], %[[RHS]] : f32
// CHECK-NEXT: %[[SELECT:.*]] = arith.select %[[CMP]], %[[LHS]], %[[RHS]] : f32
// CHECK-NEXT: %[[IS_NAN:.*]] = arith.cmpf uno, %[[LHS]], %[[LHS]] : f32
// CHECK-NEXT: %[[RESULT:.*]] = arith.select %[[IS_NAN]], %[[RHS]], %[[SELECT]] : f32
// CHECK-NEXT: return %[[RESULT]] : f32

// -----

func.func @truncf_f32(%arg0 : f32) -> bf16 {
    %0 = arith.truncf %arg0 : f32 to bf16
    return %0 : bf16
}

// CHECK-LABEL: @truncf_f32
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : i32
// CHECK-DAG: %[[C16:.+]] = arith.constant 16 : i32
// CHECK-DAG: %[[C7FC0_i16:.+]] = arith.constant 32704 : i16
// CHECK-DAG: %[[C7FFF:.+]] = arith.constant 32767 : i32
// CHECK-DAG: %[[ISNAN:.+]] = arith.cmpf une, %arg0, %arg0 : f32
// CHECK-DAG: %[[BITCAST:.+]] = arith.bitcast %arg0 : f32 to i32
// CHECK-DAG: %[[SHRUI:.+]] = arith.shrui %[[BITCAST]], %[[C16]] : i32
// CHECK-DAG: %[[BIT16:.+]] = arith.andi %[[SHRUI]], %[[C1]] : i32
// CHECK-DAG: %[[ROUNDING_BIAS:.+]] = arith.addi %[[BIT16]], %[[C7FFF]] : i32
// CHECK-DAG: %[[BIASED:.+]] = arith.addi %[[BITCAST]], %[[ROUNDING_BIAS]] : i32
// CHECK-DAG: %[[BIASED_SHIFTED:.+]] = arith.shrui %[[BIASED]], %[[C16]] : i32
// CHECK-DAG: %[[NORMAL_CASE_RESULT_i16:.+]] = arith.trunci %[[BIASED_SHIFTED]] : i32 to i16
// CHECK-DAG: %[[SELECT:.+]] = arith.select %[[ISNAN]], %[[C7FC0_i16]], %[[NORMAL_CASE_RESULT_i16]] : i16
// CHECK-DAG: %[[RESULT:.+]] = arith.bitcast %[[SELECT]] : i16 to bf16
// CHECK: return %[[RESULT]]

// -----

func.func @truncf_vector_f32(%arg0 : vector<4xf32>) -> vector<4xbf16> {
    %0 = arith.truncf %arg0 : vector<4xf32> to vector<4xbf16>
    return %0 : vector<4xbf16>
}

// CHECK-LABEL: @truncf_vector_f32
// CHECK-NOT: arith.truncf

// -----
func.func @truncf_f32_to_f8E8M0FNU(%arg0 : f32) -> f8E8M0FNU {
    %0 = arith.truncf %arg0 : f32 to f8E8M0FNU
    return %0 : f8E8M0FNU
}
// CHECK-LABLE: @truncf_f32_to_f8E8M0FNU
// CHECK: %[[BITCAST:.+]] = arith.bitcast %arg0 : f32 to i32
// CHECK: %[[C23_i32:.+]] = arith.constant 23 : i32
// CHECK: %[[SHRUI:.+]] = arith.shrui %[[BITCAST]], %[[C23_i32]] : i32
// CHECK: %[[TRUNCI:.+]] = arith.trunci %[[SHRUI]] : i32 to i8
// CHECK: %[[RESULT:.+]] = arith.bitcast %[[TRUNCI]] : i8 to f8E8M0FNU
// CHECK: return %[[RESULT]]

// -----

func.func @truncf_f16_to_f8E8M0FNU(%arg0 : f16) -> f8E8M0FNU {
    %0 = arith.truncf %arg0 : f16 to f8E8M0FNU
    return %0 : f8E8M0FNU
}
// CHECK-LABLE: @truncf_f16_to_f8E8M0FNU
// CHECK: %[[EXTF:.+]] = arith.extf %arg0 : f16 to f32
// CHECK: %[[BITCAST:.+]] = arith.bitcast %[[EXTF]] : f32 to i32
// CHECK: %[[C23_i32:.+]] = arith.constant 23 : i32
// CHECK: %[[SHRUI:.+]] = arith.shrui %[[BITCAST]], %[[C23_i32]] : i32
// CHECK: %[[TRUNCI:.+]] = arith.trunci %[[SHRUI]] : i32 to i8
// CHECK: %[[RESULT:.+]] = arith.bitcast %[[TRUNCI]] : i8 to f8E8M0FNU
// CHECK: return %[[RESULT]]

// -----

func.func @truncf_vector_f32_to_f8E8M0FNU(%arg0 : vector<4xf32>) -> vector<4xf8E8M0FNU> {
    %0 = arith.truncf %arg0 : vector<4xf32> to vector<4xf8E8M0FNU>
    return %0 : vector<4xf8E8M0FNU>
}

// CHECK-LABEL: @truncf_vector_f32_to_f8E8M0FNU
// CHECK-NOT: arith.truncf

// -----

func.func @truncf_vector_f16_to_f8E8M0FNU(%arg0 : vector<4xf16>) -> vector<4xf8E8M0FNU> {
    %0 = arith.truncf %arg0 : vector<4xf16> to vector<4xf8E8M0FNU>
    return %0 : vector<4xf8E8M0FNU>
}

// CHECK-LABEL: @truncf_vector_f16_to_f8E8M0FNU
// CHECK-NOT: arith.truncf

// -----

func.func @truncf_vector_bf16_to_f8E8M0FNU(%arg0 : vector<4xbf16>) -> vector<4xf8E8M0FNU> {
    %0 = arith.truncf %arg0 : vector<4xbf16> to vector<4xf8E8M0FNU>
    return %0 : vector<4xf8E8M0FNU>
}

// CHECK-LABEL: @truncf_vector_bf16_to_f8E8M0FNU
// CHECK-NOT: arith.truncf


// -----
func.func @extf_f8E8M0FNU_to_f32(%arg0 : f8E8M0FNU) -> f32 {
    %0 = arith.extf %arg0 : f8E8M0FNU to f32
    return %0 : f32
}

// CHECK-LABLE: @extf_f8E8M0FNU_to_f32
// CHECK: %[[BITCAST:.+]] = arith.bitcast %arg0 : f8E8M0FNU to i8
// CHECK-DAG: %[[CF8NAN:.+]] = arith.constant -1 : i8
// CHECK-DAG: %[[CF32NAN:.+]] = arith.constant -1 : i32
// CHECK-DAG: %[[C23_i32:.+]] = arith.constant 23 : i32
// CHECK: %[[EXTUI:.+]] = arith.extui %[[BITCAST]] : i8 to i32
// CHECK: %[[SHLI:.+]] = arith.shli %[[EXTUI]], %[[C23_i32]] : i32
// CHECK: %[[CMP_NAN:.+]] = arith.cmpi eq, %[[BITCAST]], %[[CF8NAN]] : i8
// CHECK: %[[SELECT_NAN:.+]] = arith.select %[[CMP_NAN]], %[[CF32NAN]], %[[SHLI]] : i32
// CHECK: %[[RESULT:.+]] = arith.bitcast %[[SELECT_NAN]] : i32 to f32
// CHECK: return %[[RESULT]]

// -----

func.func @extf_f8E8M0FNU_to_f16(%arg0 : f8E8M0FNU) -> f16 {
    %0 = arith.extf %arg0 : f8E8M0FNU to f16
    return %0 : f16
}

// CHECK-LABLE: @extf_f8E8M0FNU_to_f16
// CHECK: %[[BITCAST:.+]] = arith.bitcast %arg0 : f8E8M0FNU to i8
// CHECK-DAG: %[[CF8NAN:.+]] = arith.constant -1 : i8
// CHECK-DAG: %[[CF32NAN:.+]] = arith.constant -1 : i32
// CHECK-DAG: %[[C23_i32:.+]] = arith.constant 23 : i32
// CHECK: %[[EXTUI:.+]] = arith.extui %[[BITCAST]] : i8 to i32
// CHECK: %[[SHLI:.+]] = arith.shli %[[EXTUI]], %[[C23_i32]] : i32
// CHECK: %[[CMP_NAN:.+]] = arith.cmpi eq, %[[BITCAST]], %[[CF8NAN]] : i8
// CHECK: %[[SELECT_NAN:.+]] = arith.select %[[CMP_NAN]], %[[CF32NAN]], %[[SHLI]] : i32
// CHECK: %[[F32_RESULT:.+]] = arith.bitcast %[[SELECT_NAN]] : i32 to f32
// CHECK: %[[F16_RESULT:.+]] = arith.truncf %[[F32_RESULT]] : f32 to f16
// CHECK: return %[[F16_RESULT]]

// -----

func.func @extf_vector_f8E8M0FNU_to_f32(%arg0 : vector<4xf8E8M0FNU>) -> vector<4xf32> {
    %0 = arith.extf %arg0 : vector<4xf8E8M0FNU> to vector<4xf32>
    return %0 : vector<4xf32>
}

// CHECK-LABEL: @extf_vector_f8E8M0FNU_to_f32
// CHECK-NOT: arith.extf

// -----

func.func @extf_vector_f8E8M0FNU_to_f16(%arg0 : vector<4xf8E8M0FNU>) -> vector<4xf16> {
    %0 = arith.extf %arg0 : vector<4xf8E8M0FNU> to vector<4xf16>
    return %0 : vector<4xf16>
}

// CHECK-LABEL: @extf_vector_f8E8M0FNU_to_f16
// CHECK-NOT: arith.extf

// -----

func.func @extf_vector_f8E8M0FNU_to_bf16(%arg0 : vector<4xf8E8M0FNU>) -> vector<4xbf16> {
    %0 = arith.extf %arg0 : vector<4xf8E8M0FNU> to vector<4xbf16>
    return %0 : vector<4xbf16>
}

// CHECK-LABEL: @extf_vector_f8E8M0FNU_to_bf16
// CHECK-NOT: arith.extf


// -----

func.func @maxsi(%a: i32, %b: i32) -> i32 {
  %result = arith.maxsi %a, %b : i32
  return %result : i32
}
// CHECK-LABEL: func @maxsi
// CHECK-SAME: %[[LHS:.*]]: i32, %[[RHS:.*]]: i32
// CHECK-NEXT: %[[CMP:.*]] = arith.cmpi sgt, %[[LHS]], %[[RHS]] : i32
// CHECK-NEXT: %[[RESULT:.*]] = arith.select %[[CMP]], %[[LHS]], %[[RHS]] : i32
// CHECK-NEXT: return %[[RESULT]] : i32

// -----

func.func @minsi(%a: i32, %b: i32) -> i32 {
  %result = arith.minsi %a, %b : i32
  return %result : i32
}
// CHECK-LABEL: func @minsi
// CHECK-SAME: %[[LHS:.*]]: i32, %[[RHS:.*]]: i32
// CHECK-NEXT: %[[CMP:.*]] = arith.cmpi slt, %[[LHS]], %[[RHS]] : i32
// CHECK-NEXT: %[[RESULT:.*]] = arith.select %[[CMP]], %[[LHS]], %[[RHS]] : i32
// CHECK-NEXT: return %[[RESULT]] : i32

// -----

func.func @maxui(%a: i32, %b: i32) -> i32 {
  %result = arith.maxui %a, %b : i32
  return %result : i32
}
// CHECK-LABEL: func @maxui
// CHECK-SAME: %[[LHS:.*]]: i32, %[[RHS:.*]]: i32
// CHECK-NEXT: %[[CMP:.*]] = arith.cmpi ugt, %[[LHS]], %[[RHS]] : i32
// CHECK-NEXT: %[[RESULT:.*]] = arith.select %[[CMP]], %[[LHS]], %[[RHS]] : i32
// CHECK-NEXT: return %[[RESULT]] : i32

// -----

func.func @minui(%a: i32, %b: i32) -> i32 {
  %result = arith.minui %a, %b : i32
  return %result : i32
}
// CHECK-LABEL: func @minui
// CHECK-SAME: %[[LHS:.*]]: i32, %[[RHS:.*]]: i32
// CHECK-NEXT: %[[CMP:.*]] = arith.cmpi ult, %[[LHS]], %[[RHS]] : i32
// CHECK-NEXT: %[[RESULT:.*]] = arith.select %[[CMP]], %[[LHS]], %[[RHS]] : i32
// CHECK-NEXT: return %[[RESULT]] : i32
