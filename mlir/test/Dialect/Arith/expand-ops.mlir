// RUN: mlir-opt %s -arith-expand="include-bf16=true include-f8e8m0=true include-f4e2m1=true include-f8e5m2=true include-f8e4m3fn=true include-min-max-f=true include-min-max-i=true" -verify-diagnostics -split-input-file | FileCheck %s
// RUN: mlir-opt %s -arith-expand -split-input-file -verify-diagnostics | FileCheck %s --check-prefix=SCHECK

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
// CHECK-LABEL: @truncf_f32_to_f8E8M0FNU
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
// CHECK-LABEL: @truncf_f16_to_f8E8M0FNU
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
// CHECK: return

// -----

// The widening keeps only nnan: with contract, ExtFOp::fold would cancel it
// against a contract truncf producing %arg0, skipping its rounding.
func.func @truncf_f16_to_f8E8M0FNU_fastmath(%arg0 : f16) -> f8E8M0FNU {
    %0 = arith.truncf %arg0 fastmath<nnan,contract> : f16 to f8E8M0FNU
    return %0 : f8E8M0FNU
}
// CHECK-LABEL: @truncf_f16_to_f8E8M0FNU_fastmath
// CHECK: arith.extf %arg0 fastmath<nnan> : f16 to f32

// -----

// A rounding mode takes the generic expansion. f32's smallest normal, 2^-126,
// is above 2^-127 (code 0), so it is extended to f64 first: 896 is the f64
// exponent field of 2^-127 and 2^53 - 1 the to_nearest_even addend. Without a
// zero, the code is one below that of the same format with one, at least 0
// (what would round to zero is 2^-127, as in APFloat); 254 is max-finite
// (2^127), 255 the NaN.
func.func @truncf_f32_to_f8E8M0FNU_to_nearest_even(%arg0 : f32) -> f8E8M0FNU {
    %0 = arith.truncf %arg0 to_nearest_even : f32 to f8E8M0FNU
    return %0 : f8E8M0FNU
}
// CHECK-LABEL: @truncf_f32_to_f8E8M0FNU_to_nearest_even
// CHECK: %[[WIDE:.+]] = arith.extf %arg0 : f32 to f64
// CHECK-NEXT: arith.bitcast %[[WIDE]] : f64 to i64
// CHECK-DAG: arith.constant 896 : i64
// CHECK-DAG: arith.constant 9007199254740991 : i64
// CHECK: %[[GT:.+]] = arith.cmpi ugt, %[[C:.+]], %[[ONE:.+]] : i64
// CHECK-NEXT: %[[ATLEAST1:.+]] = arith.select %[[GT]], %[[C]], %[[ONE]] : i64
// CHECK-NEXT: %[[CODE:.+]] = arith.subi %[[ATLEAST1]], %[[ONE]] : i64
// CHECK-NEXT: %[[MAX:.+]] = arith.constant 254 : i64
// CHECK-NEXT: %[[OVF:.+]] = arith.cmpi ugt, %[[CODE]], %[[MAX]] : i64
// CHECK-NEXT: %[[NAN:.+]] = arith.constant 255 : i64
// CHECK-NEXT: %[[SEL:.+]] = arith.select %[[OVF]], %[[NAN]], %[[CODE]] : i64
// CHECK-NEXT: %[[TRUNCI:.+]] = arith.trunci %[[SEL]] : i64 to i8
// CHECK-NEXT: %[[RESULT:.+]] = arith.bitcast %[[TRUNCI]] : i8 to f8E8M0FNU
// CHECK-NEXT: return %[[RESULT]]

// -----

// to_nearest_away adds 2^53; an f16 is extended to f64 too.
func.func @truncf_f16_to_f8E8M0FNU_to_nearest_away(%arg0 : f16) -> f8E8M0FNU {
    %0 = arith.truncf %arg0 to_nearest_away : f16 to f8E8M0FNU
    return %0 : f8E8M0FNU
}
// CHECK-LABEL: @truncf_f16_to_f8E8M0FNU_to_nearest_away
// CHECK: %[[WIDE:.+]] = arith.extf %arg0 : f16 to f64
// CHECK-NEXT: arith.bitcast %[[WIDE]] : f64 to i64
// CHECK: arith.constant 9007199254740992 : i64
// CHECK: arith.constant 254 : i64
// CHECK: arith.trunci %{{.+}} : i64 to i8

// -----

// toward_zero shifts without an addend; an overflow stops at max-finite, and
// only a NaN or Inf (the f64 bits 0x7FF0000000000000 and above) gives the NaN.
func.func @truncf_f64_to_f8E8M0FNU_toward_zero(%arg0 : f64) -> f8E8M0FNU {
    %0 = arith.truncf %arg0 toward_zero : f64 to f8E8M0FNU
    return %0 : f8E8M0FNU
}
// CHECK-LABEL: @truncf_f64_to_f8E8M0FNU_toward_zero
// CHECK: arith.bitcast %arg0 : f64 to i64
// CHECK-NOT: arith.extf
// CHECK: %[[V:.+]] = arith.shli %{{.+}}, %{{.+}} : i64
// CHECK-NEXT: %[[C54:.+]] = arith.constant 54 : i64
// CHECK-NEXT: arith.shrui %[[V]], %[[C54]] : i64
// CHECK: %[[GT:.+]] = arith.cmpi ugt, %[[C:.+]], %[[ONE:.+]] : i64
// CHECK-NEXT: %[[ATLEAST1:.+]] = arith.select %[[GT]], %[[C]], %[[ONE]] : i64
// CHECK-NEXT: %[[CODE:.+]] = arith.subi %[[ATLEAST1]], %[[ONE]] : i64
// CHECK-NEXT: %[[MAX:.+]] = arith.constant 254 : i64
// CHECK-NEXT: %[[LT:.+]] = arith.cmpi ult, %[[CODE]], %[[MAX]] : i64
// CHECK-NEXT: %[[CLAMPED:.+]] = arith.select %[[LT]], %[[CODE]], %[[MAX]] : i64
// CHECK-NEXT: %[[INF:.+]] = arith.constant 9218868437227405312 : i64
// CHECK-NEXT: %[[ISNAN:.+]] = arith.cmpi uge, %{{.+}}, %[[INF]] : i64
// CHECK-NEXT: %[[NAN:.+]] = arith.constant 255 : i64
// CHECK-NEXT: arith.select %[[ISNAN]], %[[NAN]], %[[CLAMPED]] : i64

// -----

// The format has no sign, yet upward rounds x as APFloat does, with its sign:
// the magnitude rounds up (2^54 - 1 added) only for x >= 0, and only then does
// an overflow give the NaN.
func.func @truncf_vector_f32_to_f8E8M0FNU_upward(%arg0 : vector<4xf32>)
    -> vector<4xf8E8M0FNU> {
    %0 = arith.truncf %arg0 upward : vector<4xf32> to vector<4xf8E8M0FNU>
    return %0 : vector<4xf8E8M0FNU>
}
// CHECK-LABEL: @truncf_vector_f32_to_f8E8M0FNU_upward
// CHECK: %[[WIDE:.+]] = arith.extf %arg0 : vector<4xf32> to vector<4xf64>
// CHECK-NEXT: %[[BITS:.+]] = arith.bitcast %[[WIDE]] : vector<4xf64> to vector<4xi64>
// CHECK-NEXT: %[[ZERO:.+]] = arith.constant dense<0> : vector<4xi64>
// CHECK-NEXT: %[[UP:.+]] = arith.cmpi sge, %[[BITS]], %[[ZERO]] : vector<4xi64>
// CHECK: %[[ALL:.+]] = arith.constant dense<18014398509481983> : vector<4xi64>
// CHECK-NEXT: arith.select %[[UP]], %[[ALL]], %[[ZERO]] : vector<4xi1>, vector<4xi64>
// CHECK: %[[MAX:.+]] = arith.constant dense<254> : vector<4xi64>
// CHECK: %[[OVF:.+]] = arith.cmpi ugt, %{{.+}}, %[[MAX]] : vector<4xi64>
// CHECK-NEXT: arith.andi %[[OVF]], %[[UP]] : vector<4xi1>
// CHECK: arith.trunci %{{.+}} : vector<4xi64> to vector<4xi8>

// -----

// downward rounds the magnitude up for x < 0; such an overflow would be a
// negative NaN, which the format lacks (APFloat asserts), so it saturates.
func.func @truncf_f64_to_f8E8M0FNU_downward(%arg0 : f64) -> f8E8M0FNU {
    %0 = arith.truncf %arg0 downward : f64 to f8E8M0FNU
    return %0 : f8E8M0FNU
}
// CHECK-LABEL: @truncf_f64_to_f8E8M0FNU_downward
// CHECK: %[[BITS:.+]] = arith.bitcast %arg0 : f64 to i64
// CHECK-NEXT: %[[ZERO:.+]] = arith.constant 0 : i64
// CHECK-NEXT: %[[UP:.+]] = arith.cmpi slt, %[[BITS]], %[[ZERO]] : i64
// CHECK: arith.select %[[UP]], %{{.+}}, %[[ZERO]] : i64
// CHECK: %[[GT:.+]] = arith.cmpi ugt, %[[C:.+]], %[[ONE:.+]] : i64
// CHECK-NEXT: %[[ATLEAST1:.+]] = arith.select %[[GT]], %[[C]], %[[ONE]] : i64
// CHECK-NEXT: %[[CODE:.+]] = arith.subi %[[ATLEAST1]], %[[ONE]] : i64
// CHECK-NEXT: %[[MAX:.+]] = arith.constant 254 : i64
// CHECK-NEXT: %[[LT:.+]] = arith.cmpi ult, %[[CODE]], %[[MAX]] : i64
// CHECK-NEXT: %[[CLAMPED:.+]] = arith.select %[[LT]], %[[CODE]], %[[MAX]] : i64
// CHECK-NEXT: %[[INF:.+]] = arith.constant 9218868437227405312 : i64
// CHECK-NEXT: %[[ISNAN:.+]] = arith.cmpi uge, %{{.+}}, %[[INF]] : i64
// CHECK-NEXT: %[[NAN:.+]] = arith.constant 255 : i64
// CHECK-NEXT: arith.select %[[ISNAN]], %[[NAN]], %[[CLAMPED]] : i64

// -----

// Without a mode, a source wider than f32 rounds toward zero once, from its
// own bits (an f128 rounded to odd into f64 first), like the exponent of an
// f32 value.
func.func @truncf_wide_to_f8E8M0FNU(%arg0 : f64, %arg1 : f128)
    -> (f8E8M0FNU, f8E8M0FNU) {
    %0 = arith.truncf %arg0 : f64 to f8E8M0FNU
    %1 = arith.truncf %arg1 : f128 to f8E8M0FNU
    return %0, %1 : f8E8M0FNU, f8E8M0FNU
}
// CHECK-LABEL: @truncf_wide_to_f8E8M0FNU
// CHECK-NOT: arith.truncf %arg0
// CHECK: arith.bitcast %arg0 : f64 to i64
// CHECK: %[[V:.+]] = arith.shli %{{.+}}, %{{.+}} : i64
// CHECK-NEXT: %[[C54:.+]] = arith.constant 54 : i64
// CHECK-NEXT: arith.shrui %[[V]], %[[C54]] : i64
// CHECK: arith.constant 9218868437227405312 : i64
// CHECK: %[[R0:.+]] = arith.bitcast %{{.+}} : i8 to f8E8M0FNU
// CHECK: %[[R:.+]] = arith.truncf %arg1 : f128 to f64
// CHECK-NEXT: arith.extf %[[R]] : f64 to f128
// CHECK: %[[R1:.+]] = arith.bitcast %{{.+}} : i8 to f8E8M0FNU
// CHECK-NEXT: return %[[R0]], %[[R1]]

// -----

func.func @scaling_truncf_f32_to_f4E2M1FN(%arg0 : f32, %arg1: f8E8M0FNU) -> f4E2M1FN {
    %0 = arith.scaling_truncf %arg0, %arg1 : f32, f8E8M0FNU to f4E2M1FN
    return %0 : f4E2M1FN
}

// SCHECK-LABEL: @scaling_truncf_f32_to_f4E2M1FN
// SCHECK: %[[SCALEF32:.+]] = arith.extf %arg1 : f8E8M0FNU to f32
// SCHECK: %[[DIVF:.+]] = arith.divf %arg0, %[[SCALEF32]] : f32
// SCHECK: %[[RESULT:.+]] = arith.truncf %[[DIVF]] : f32 to f4E2M1FN
// SCHECK: return %[[RESULT]]

// -----

func.func @scaling_truncf_vector_f16_to_f6E3M2FN(%arg0 : vector<4xf16>, %arg1: vector<4xf8E8M0FNU>) -> vector<4xf6E3M2FN> {
    %0 = arith.scaling_truncf %arg0, %arg1 : vector<4xf16>, vector<4xf8E8M0FNU> to vector<4xf6E3M2FN>
    return %0 : vector<4xf6E3M2FN>
}

// SCHECK-LABEL: @scaling_truncf_vector_f16_to_f6E3M2FN
// SCHECK: %[[SCALEF16:.+]] = arith.extf %arg1 : vector<4xf8E8M0FNU> to vector<4xf16>
// SCHECK: %[[DIVF:.+]] = arith.divf %arg0, %[[SCALEF16]] : vector<4xf16>
// SCHECK: %[[RESULT:.+]] = arith.truncf %[[DIVF]] : vector<4xf16> to vector<4xf6E3M2FN>
// SCHECK: return %[[RESULT]] : vector<4xf6E3M2FN>

// -----

func.func @scaling_truncf_propagate_rounding_mode_fast_math(%arg0 : vector<4xf16>, %arg1: vector<4xf16>) -> vector<4xf6E3M2FN> {
    %0 = arith.scaling_truncf %arg0, %arg1 to_nearest_even fastmath<fast> : vector<4xf16>, vector<4xf16> to vector<4xf6E3M2FN>
    return %0 : vector<4xf6E3M2FN>
}
// SCHECK-LABEL: @scaling_truncf_propagate_rounding_mode_fast_math
// SCHECK: %[[SCALEF8:.+]] = arith.truncf %arg1 fastmath<fast> : vector<4xf16> to vector<4xf8E8M0FNU>
// SCHECK: %[[SCALEINTY:.+]] = arith.extf %[[SCALEF8]] fastmath<fast> : vector<4xf8E8M0FNU> to vector<4xf16>
// SCHECK: %[[DIVF:.+]] = arith.divf %arg0, %[[SCALEINTY]] fastmath<fast> : vector<4xf16>
// SCHECK: %[[TRUNCF:.+]] = arith.truncf [[_:%[a-zA-Z0-9_]+]] to_nearest_even fastmath<fast> : vector<4xf16> to vector<4xf6E3M2FN>
// SCHECK: return %[[TRUNCF]] : vector<4xf6E3M2FN>

// -----

func.func @scaling_truncf_f16_to_f4E2M1FN_using_f16_scales(%arg0: f16, %arg1 : f16) -> f4E2M1FN {
    %0 = arith.scaling_truncf %arg0, %arg1 : f16, f16 to f4E2M1FN
    return %0 : f4E2M1FN
}
// SCHECK-LABEL: @scaling_truncf_f16_to_f4E2M1FN_using_f16_scales
// SCHECK: %[[SCALETRUNCF:.+]] = arith.truncf %arg1 : f16 to f8E8M0FN
// SCHECK: return

// -----
func.func @scaling_truncf_vector_f16_to_f4E2M1FN_using_f16_scales(%arg0: vector<4xf16>, %arg1 : vector<4xf16>) -> vector<4xf4E2M1FN> {
    %0 = arith.scaling_truncf %arg0, %arg1 : vector<4xf16>, vector<4xf16> to vector<4xf4E2M1FN>
    return %0 : vector<4xf4E2M1FN>
}
// SCHECK-LABEL: @scaling_truncf_vector_f16_to_f4E2M1FN_using_f16_scales
// SCHECK: %[[SCALETRUNCF:.+]] = arith.truncf %arg1 : vector<4xf16> to vector<4xf8E8M0FNU>
// SCHECK: return

// -----

func.func @invalid_scaling_truncf_to_f4E2M1FN(%arg0: f16, %arg1 : f8E5M2FNUZ) -> f4E2M1FN {
    // expected-error@+1 {{failed to legalize operation 'arith.scaling_truncf' that was explicitly marked illegal}}
    %0 = arith.scaling_truncf %arg0, %arg1 : f16, f8E5M2FNUZ to f4E2M1FN
    return %0 : f4E2M1FN
}

// -----

// 0x00 decodes to 2^-127 (the f32 subnormal 4194304) and the NaN 0xFF to the
// quiet NaN 0x7FC00000 (2143289344), as the constant folder does.
func.func @extf_f8E8M0FNU_to_f32(%arg0 : f8E8M0FNU) -> f32 {
    %0 = arith.extf %arg0 : f8E8M0FNU to f32
    return %0 : f32
}

// CHECK-LABEL: @extf_f8E8M0FNU_to_f32
// CHECK: %[[BITCAST:.+]] = arith.bitcast %arg0 : f8E8M0FNU to i8
// CHECK: %[[C23_i32:.+]] = arith.constant 23 : i32
// CHECK: %[[EXTUI:.+]] = arith.extui %[[BITCAST]] : i8 to i32
// CHECK: %[[SHLI:.+]] = arith.shli %[[EXTUI]], %[[C23_i32]] : i32
// CHECK-DAG: %[[C0_i8:.+]] = arith.constant 0 : i8
// CHECK-DAG: %[[CMIN_i32:.+]] = arith.constant 4194304 : i32
// CHECK: %[[IS_MIN:.+]] = arith.cmpi eq, %[[BITCAST]], %[[C0_i8]] : i8
// CHECK: %[[SELECT_MIN:.+]] = arith.select %[[IS_MIN]], %[[CMIN_i32]], %[[SHLI]] : i32
// CHECK-DAG: %[[CF8NAN:.+]] = arith.constant -1 : i8
// CHECK-DAG: %[[CF32NAN:.+]] = arith.constant 2143289344 : i32
// CHECK: %[[CMP_NAN:.+]] = arith.cmpi eq, %[[BITCAST]], %[[CF8NAN]] : i8
// CHECK: %[[SELECT_NAN:.+]] = arith.select %[[CMP_NAN]], %[[CF32NAN]], %[[SELECT_MIN]] : i32
// CHECK: %[[RESULT:.+]] = arith.bitcast %[[SELECT_NAN]] : i32 to f32
// CHECK: return %[[RESULT]]

// -----

func.func @extf_f8E8M0FNU_to_f32_no_nan(%arg0 : f8E8M0FNU) -> f32 {
    %0 = arith.extf %arg0 fastmath<nnan> : f8E8M0FNU to f32
    return %0 : f32
}

// CHECK-LABEL: @extf_f8E8M0FNU_to_f32_no_nan
// CHECK: %[[BITCAST:.+]] = arith.bitcast %arg0 : f8E8M0FNU to i8
// CHECK: %[[C23_i32:.+]] = arith.constant 23 : i32
// CHECK: %[[EXTUI:.+]] = arith.extui %[[BITCAST]] : i8 to i32
// CHECK: %[[SHLI:.+]] = arith.shli %[[EXTUI]], %[[C23_i32]] : i32
// CHECK-DAG: %[[C0_i8:.+]] = arith.constant 0 : i8
// CHECK-DAG: %[[CMIN_i32:.+]] = arith.constant 4194304 : i32
// CHECK: %[[IS_MIN:.+]] = arith.cmpi eq, %[[BITCAST]], %[[C0_i8]] : i8
// CHECK: %[[SELECT_MIN:.+]] = arith.select %[[IS_MIN]], %[[CMIN_i32]], %[[SHLI]] : i32
// CHECK: %[[RESULT:.+]] = arith.bitcast %[[SELECT_MIN]] : i32 to f32
// CHECK: return %[[RESULT]]

// -----

func.func @extf_f8E8M0FNU_to_f16(%arg0 : f8E8M0FNU) -> f16 {
    %0 = arith.extf %arg0 : f8E8M0FNU to f16
    return %0 : f16
}

// CHECK-LABEL: @extf_f8E8M0FNU_to_f16
// CHECK: %[[BITCAST:.+]] = arith.bitcast %arg0 : f8E8M0FNU to i8
// CHECK-DAG: %[[C23_i32:.+]] = arith.constant 23 : i32
// CHECK: %[[EXTUI:.+]] = arith.extui %[[BITCAST]] : i8 to i32
// CHECK: %[[SHLI:.+]] = arith.shli %[[EXTUI]], %[[C23_i32]] : i32
// CHECK-DAG: %[[C0_i8:.+]] = arith.constant 0 : i8
// CHECK-DAG: %[[CMIN_i32:.+]] = arith.constant 4194304 : i32
// CHECK: %[[IS_MIN:.+]] = arith.cmpi eq, %[[BITCAST]], %[[C0_i8]] : i8
// CHECK: %[[SELECT_MIN:.+]] = arith.select %[[IS_MIN]], %[[CMIN_i32]], %[[SHLI]] : i32
// CHECK-DAG: %[[CF8NAN:.+]] = arith.constant -1 : i8
// CHECK-DAG: %[[CF32NAN:.+]] = arith.constant 2143289344 : i32
// CHECK: %[[CMP_NAN:.+]] = arith.cmpi eq, %[[BITCAST]], %[[CF8NAN]] : i8
// CHECK: %[[SELECT_NAN:.+]] = arith.select %[[CMP_NAN]], %[[CF32NAN]], %[[SELECT_MIN]] : i32
// CHECK: %[[F32_RESULT:.+]] = arith.bitcast %[[SELECT_NAN]] : i32 to f32
// CHECK: %[[F16_RESULT:.+]] = arith.truncf %[[F32_RESULT]] : f32 to f16
// CHECK: return %[[F16_RESULT]]

// -----

func.func @extf_f8E8M0FNU_to_bf16(%arg0 : f8E8M0FNU) -> bf16 {
    %0 = arith.extf %arg0 : f8E8M0FNU to bf16
    return %0 : bf16
}

// bf16 takes the high half of the f32 bits (no truncf): 0x00 gives 0x0040
// (2^-127, f32 4194304) and the NaN 0xFF gives 0x7FC0 (f32 2143289344).
// CHECK-LABEL: @extf_f8E8M0FNU_to_bf16
// CHECK-DAG: arith.constant 4194304 : i32
// CHECK: arith.cmpi eq, %{{.+}}, %{{.+}} : i8
// CHECK: arith.select %{{.+}}, %{{.+}}, %{{.+}} : i32
// CHECK-DAG: arith.constant 2143289344 : i32
// CHECK: %[[SELECT_NAN:.+]] = arith.select %{{.+}}, %{{.+}}, %{{.+}} : i32
// CHECK: %[[C16:.+]] = arith.constant 16 : i32
// CHECK: %[[HIGH:.+]] = arith.shrui %[[SELECT_NAN]], %[[C16]] : i32
// CHECK: %[[BITS:.+]] = arith.trunci %[[HIGH]] : i32 to i16
// CHECK: %[[RESULT:.+]] = arith.bitcast %[[BITS]] : i16 to bf16
// CHECK-NOT: arith.truncf
// CHECK: return %[[RESULT]]

// -----

func.func @extf_f8E8M0FNU_to_f64(%arg0 : f8E8M0FNU) -> f64 {
    %0 = arith.extf %arg0 : f8E8M0FNU to f64
    return %0 : f64
}

// A finite code c is the normal f64 2^(c - 127), built as (c + 896) << 52; the
// NaN 0xFF gives 0x7FF8000000000000 (9221120237041090560).
// CHECK-LABEL: @extf_f8E8M0FNU_to_f64
// CHECK: %[[BITCAST:.+]] = arith.bitcast %arg0 : f8E8M0FNU to i8
// CHECK: %[[EXTUI:.+]] = arith.extui %[[BITCAST]] : i8 to i64
// CHECK: %[[C896:.+]] = arith.constant 896 : i64
// CHECK: %[[EXP:.+]] = arith.addi %[[EXTUI]], %[[C896]] : i64
// CHECK: %[[C52:.+]] = arith.constant 52 : i64
// CHECK: %[[BITS:.+]] = arith.shli %[[EXP]], %[[C52]] : i64
// CHECK-DAG: %[[CF8NAN:.+]] = arith.constant -1 : i8
// CHECK-DAG: %[[CF64NAN:.+]] = arith.constant 9221120237041090560 : i64
// CHECK: %[[IS_NAN:.+]] = arith.cmpi eq, %[[BITCAST]], %[[CF8NAN]] : i8
// CHECK: %[[SELECT:.+]] = arith.select %[[IS_NAN]], %[[CF64NAN]], %[[BITS]] : i64
// CHECK-NOT: arith.extf
// CHECK: %[[RESULT:.+]] = arith.bitcast %[[SELECT]] : i64 to f64
// CHECK-NOT: arith.extf
// CHECK: return %[[RESULT]]

// -----

func.func @extf_f8E8M0FNU_to_f128(%arg0 : f8E8M0FNU) -> f128 {
    %0 = arith.extf %arg0 : f8E8M0FNU to f128
    return %0 : f128
}

// A wider destination extends those f64 bits exactly.
// CHECK-LABEL: @extf_f8E8M0FNU_to_f128
// CHECK: arith.extui %{{.+}} : i8 to i64
// CHECK: %[[F64:.+]] = arith.bitcast %{{.+}} : i64 to f64
// CHECK: %[[RESULT:.+]] = arith.extf %[[F64]] : f64 to f128
// CHECK: return %[[RESULT]]

// -----

func.func @extf_f8E8M0FNU_to_f64_no_nan(%arg0 : f8E8M0FNU) -> f64 {
    %0 = arith.extf %arg0 fastmath<nnan> : f8E8M0FNU to f64
    return %0 : f64
}

// Without the NaN select, 0xFF gives (255 + 896) << 52 = 2^128.
// CHECK-LABEL: @extf_f8E8M0FNU_to_f64_no_nan
// CHECK: %[[EXTUI:.+]] = arith.extui %{{.+}} : i8 to i64
// CHECK: %[[EXP:.+]] = arith.addi %[[EXTUI]], %{{.+}} : i64
// CHECK: %[[BITS:.+]] = arith.shli %[[EXP]], %{{.+}} : i64
// CHECK-NEXT: %[[RESULT:.+]] = arith.bitcast %[[BITS]] : i64 to f64
// CHECK-NEXT: return %[[RESULT]]

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
// CHECK: return

// -----

func.func @scaling_extf_to_f32(%arg0: f4E2M1FN, %arg1 : f8E8M0FNU) -> f32 {
    %0 = arith.scaling_extf %arg0, %arg1 : f4E2M1FN, f8E8M0FNU to f32
    return %0 : f32 
}

// SCHECK-LABEL: @scaling_extf_to_f32
// SCHECK: %[[EXT_SCALE:.+]] = arith.extf %arg1 : f8E8M0FNU to f32
// SCHECK: %[[EXT_INPUT:.+]] = arith.extf %arg0 : f4E2M1FN to f32
// SCHECK: %[[RESULT:.+]] = arith.mulf %[[EXT_INPUT]], %[[EXT_SCALE]] : f32
// SCHECK: return %[[RESULT]]

// -----

func.func @scaling_extf_to_f32_using_f16_scales(%arg0: f4E2M1FN, %arg1 : f16) -> f32 {
    %0 = arith.scaling_extf %arg0, %arg1 : f4E2M1FN, f16 to f32
    return %0 : f32 
}

// SCHECK-LABEL: @scaling_extf_to_f32_using_f16_scales
// SCHECK: %[[TRUNCF_SCALE:.+]] = arith.truncf %arg1 : f16 to f8E8M0FNU
// SCHECK: %[[EXT_SCALE:.+]] = arith.extf %[[TRUNCF_SCALE]] : f8E8M0FNU to f32
// SCHECK: %[[EXT_INPUT:.+]] = arith.extf %arg0 : f4E2M1FN to f32
// SCHECK: %[[RESULT:.+]] = arith.mulf %[[EXT_INPUT]], %[[EXT_SCALE]] : f32
// SCHECK: return %[[RESULT]]

// -----

func.func @invalid_scaling_extf_to_f32(%arg0: f4E2M1FN, %arg1 : f8E5M2FNUZ) -> f32 {
    // expected-error@+1 {{failed to legalize operation 'arith.scaling_extf' that was explicitly marked illegal}}
    %0 = arith.scaling_extf %arg0, %arg1 : f4E2M1FN, f8E5M2FNUZ to f32
    return %0 : f32
}

// -----

func.func @scaling_extf_vector_to_f32(%arg0: vector<4xf4E2M1FN>, %arg1 : vector<4xf8E8M0FNU>) -> vector<4xf32> {
    %0 = arith.scaling_extf %arg0, %arg1 : vector<4xf4E2M1FN>, vector<4xf8E8M0FNU> to vector<4xf32>
    return %0 : vector<4xf32>
}

// SCHECK-LABEL: @scaling_extf_vector_to_f32
// SCHECK: %[[EXT_SCALE:.+]] = arith.extf %arg1 : vector<4xf8E8M0FNU> to vector<4xf32>
// SCHECK: %[[EXT_INPUT:.+]] = arith.extf %arg0 : vector<4xf4E2M1FN> to vector<4xf32>
// SCHECK: %[[RESULT:.+]] = arith.mulf %[[EXT_INPUT]], %[[EXT_SCALE]] : vector<4xf32> 
// SCHECK: return %[[RESULT]]

// -----

func.func @scaling_extf_vector_to_f16(%arg0: vector<4xf4E2M1FN>, %arg1 : vector<4xf8E8M0FNU>) -> vector<4xf16> {
    %0 = arith.scaling_extf %arg0, %arg1 : vector<4xf4E2M1FN>, vector<4xf8E8M0FNU> to vector<4xf16>
    return %0 : vector<4xf16>
}

// SCHECK-LABEL: @scaling_extf_vector_to_f16
// SCHECK: %[[EXT_SCALE:.+]] = arith.extf %arg1 : vector<4xf8E8M0FNU> to vector<4xf16>
// SCHECK: %[[EXT_INPUT:.+]] = arith.extf %arg0 : vector<4xf4E2M1FN> to vector<4xf16>
// SCHECK: %[[RESULT:.+]] = arith.mulf %[[EXT_INPUT]], %[[EXT_SCALE]] : vector<4xf16> 
// SCHECK: return %[[RESULT]]

// -----

func.func @scaling_extf_vector_to_bf16(%arg0: vector<4xf4E2M1FN>, %arg1 : vector<4xf8E8M0FNU>) -> vector<4xbf16> {
    %0 = arith.scaling_extf %arg0, %arg1 : vector<4xf4E2M1FN>, vector<4xf8E8M0FNU> to vector<4xbf16>
    return %0 : vector<4xbf16>
}

// SCHECK-LABEL: @scaling_extf_vector_to_bf16
// SCHECK: %[[EXT_SCALE:.+]] = arith.extf %arg1 : vector<4xf8E8M0FNU> to vector<4xbf16>
// SCHECK: %[[EXT_INPUT:.+]] = arith.extf %arg0 : vector<4xf4E2M1FN> to vector<4xbf16>
// SCHECK: %[[RESULT:.+]] = arith.mulf %[[EXT_INPUT]], %[[EXT_SCALE]] : vector<4xbf16> 
// SCHECK: return %[[RESULT]]

// -----

func.func @scaling_extf_vector_to_f32_using_f16_scales(%arg0: vector<4xf4E2M1FN>, %arg1 : vector<4xf16>) -> vector<4xf32> {
    %0 = arith.scaling_extf %arg0, %arg1 : vector<4xf4E2M1FN>, vector<4xf16> to vector<4xf32>
    return %0 : vector<4xf32>
}

// SCHECK-LABEL: @scaling_extf_vector_to_f32_using_f16_scales
// SCHECK: %[[TRUNCF_SCALE:.+]] = arith.truncf %arg1 : vector<4xf16> to vector<4xf8E8M0FNU>
// SCHECK: %[[EXT_SCALE:.+]] = arith.extf %[[TRUNCF_SCALE]] : vector<4xf8E8M0FNU> to vector<4xf32>
// SCHECK: %[[EXT_INPUT:.+]] = arith.extf %arg0 : vector<4xf4E2M1FN> to vector<4xf32>
// SCHECK: %[[RESULT:.+]] = arith.mulf %[[EXT_INPUT]], %[[EXT_SCALE]] : vector<4xf32>
// SCHECK: return %[[RESULT]]

// -----

func.func @scaling_extf_vector_to_f32_using_f16_scales_fastmath(%arg0: vector<4xf4E2M1FN>, %arg1 : vector<4xf16>) -> vector<4xf32> {
    %0 = arith.scaling_extf %arg0, %arg1 fastmath<fast> : vector<4xf4E2M1FN>, vector<4xf16> to vector<4xf32>
    return %0 : vector<4xf32>
}

// SCHECK-LABEL: @scaling_extf_vector_to_f32_using_f16_scales_fastmath
// SCHECK: %[[TRUNCF_SCALE:.+]] = arith.truncf %arg1 fastmath<fast> : vector<4xf16> to vector<4xf8E8M0FNU>
// SCHECK: %[[EXT_SCALE:.+]] = arith.extf %[[TRUNCF_SCALE]] fastmath<fast> : vector<4xf8E8M0FNU> to vector<4xf32>
// SCHECK: %[[EXT_INPUT:.+]] = arith.extf %arg0 fastmath<fast> : vector<4xf4E2M1FN> to vector<4xf32>
// SCHECK: %[[RESULT:.+]] = arith.mulf %[[EXT_INPUT]], %[[EXT_SCALE]] fastmath<fast> : vector<4xf32>
// SCHECK: return %[[RESULT]]

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

// -----

func.func @truncf_f32_to_f4E2M1FN(%arg0 : f32) -> f4E2M1FN {
    %0 = arith.truncf %arg0 : f32 to f4E2M1FN
    return %0 : f4E2M1FN
}

// CHECK-LABEL: @truncf_f32_to_f4E2M1FN
// CHECK-NOT: arith.truncf

// -----

func.func @truncf_vector_f32_to_f4E2M1FN(%arg0 : vector<4xf32>) -> vector<4xf4E2M1FN> {
    %0 = arith.truncf %arg0 : vector<4xf32> to vector<4xf4E2M1FN>
    return %0 : vector<4xf4E2M1FN>
}

// CHECK-LABEL: @truncf_vector_f32_to_f4E2M1FN
// CHECK-NOT: arith.truncf

// -----

func.func @extf_f4E2M1FN_to_f32(%arg0 : f4E2M1FN) -> f32 {
    %0 = arith.extf %arg0 : f4E2M1FN to f32
    return %0 : f32
}

// CHECK-LABEL: @extf_f4E2M1FN_to_f32
// CHECK-NOT: arith.extf

// -----

func.func @extf_vector_f4E2M1FN_to_f32(%arg0 : vector<4xf4E2M1FN>) -> vector<4xf32> {
    %0 = arith.extf %arg0 : vector<4xf4E2M1FN> to vector<4xf32>
    return %0 : vector<4xf32>
}

// CHECK-LABEL: @extf_vector_f4E2M1FN_to_f32
// CHECK-NOT: arith.extf

// -----

func.func @extf_f8E5M2_to_f32(%arg0 : f8E5M2) -> f32 {
    %0 = arith.extf %arg0 : f8E5M2 to f32
    return %0 : f32
}

// f8E5M2 has f16's exponent layout: the code is shifted into an f16, which one
// exact arith.extf takes to f32.
// CHECK-LABEL: @extf_f8E5M2_to_f32
// CHECK-SAME: %[[ARG0:.+]]: f8E5M2
// CHECK: %[[BITCAST:.+]] = arith.bitcast %[[ARG0]] : f8E5M2 to i8
// CHECK: %[[EXTUI:.+]] = arith.extui %[[BITCAST]] : i8 to i32
// CHECK-NOT: arith.extf
// CHECK: %[[C8:.+]] = arith.constant 8 : i32
// CHECK: arith.shli %{{.+}}, %[[C8]] : i32
// CHECK-NOT: arith.extf
// CHECK: %[[BITS:.+]] = arith.trunci %{{.+}} : i32 to i16
// CHECK: %[[HALF:.+]] = arith.bitcast %[[BITS]] : i16 to f16
// CHECK: %[[RESULT:.+]] = arith.extf %[[HALF]] : f16 to f32
// CHECK: return %[[RESULT]]

// -----

// f8E5M2 is the high byte of an f16, so the decode is a shift by 8; a NaN also
// gets the quiet bit.
func.func @extf_f8E5M2_to_f16(%arg0 : f8E5M2) -> f16 {
    %0 = arith.extf %arg0 : f8E5M2 to f16
    return %0 : f16
}

// CHECK-LABEL: @extf_f8E5M2_to_f16
// CHECK: %[[BITCAST:.+]] = arith.bitcast %arg0 : f8E5M2 to i8
// CHECK: %[[EXTUI:.+]] = arith.extui %[[BITCAST]] : i8 to i32
// CHECK-NOT: arith.extf
// CHECK: %[[C8:.+]] = arith.constant 8 : i32
// CHECK: %[[SHIFTED:.+]] = arith.shli %{{.+}}, %[[C8]] : i32
// CHECK: %[[QUIET:.+]] = arith.ori %[[SHIFTED]], %{{.+}} : i32
// CHECK: %[[CODE:.+]] = arith.select %{{.+}}, %[[QUIET]], %[[SHIFTED]] : i32
// CHECK-NOT: arith.extf
// CHECK: %[[BITS:.+]] = arith.trunci %[[CODE]] : i32 to i16
// CHECK: %[[RESULT:.+]] = arith.bitcast %[[BITS]] : i16 to f16
// CHECK: return %[[RESULT]]

// -----

func.func @extf_vector_f8E5M2_to_f32(%arg0 : vector<4xf8E5M2>) -> vector<4xf32> {
    %0 = arith.extf %arg0 : vector<4xf8E5M2> to vector<4xf32>
    return %0 : vector<4xf32>
}

// CHECK-LABEL: @extf_vector_f8E5M2_to_f32
// CHECK: arith.bitcast %arg0 : vector<4xf8E5M2> to vector<4xi8>
// CHECK: arith.extui %{{.+}} : vector<4xi8> to vector<4xi32>
// CHECK-NOT: arith.extf
// CHECK: arith.trunci %{{.+}} : vector<4xi32> to vector<4xi16>
// CHECK: %[[HALF:.+]] = arith.bitcast %{{.+}} : vector<4xi16> to vector<4xf16>
// CHECK: arith.extf %[[HALF]] : vector<4xf16> to vector<4xf32>
// CHECK: return

// -----

func.func @truncf_f32_to_f8E5M2(%arg0 : f32) -> f8E5M2 {
    %0 = arith.truncf %arg0 : f32 to f8E5M2
    return %0 : f8E5M2
}

// CHECK-LABEL: @truncf_f32_to_f8E5M2
// CHECK: %[[BITS:.+]] = arith.bitcast %arg0 : f32 to i32
// CHECK-NOT: arith.truncf
// One rounding from the f32 bits; an overflow gives the Inf code 124 (0x7C).
// CHECK: arith.constant 124 : i32
// CHECK-NOT: arith.truncf
// CHECK: %[[CODE:.+]] = arith.trunci %{{.+}} : i32 to i8
// CHECK: %[[RESULT:.+]] = arith.bitcast %[[CODE]] : i8 to f8E5M2
// CHECK: return %[[RESULT]]

// -----

func.func @truncf_f16_to_f8E5M2(%arg0 : f16) -> f8E5M2 {
    %0 = arith.truncf %arg0 : f16 to f8E5M2
    return %0 : f8E5M2
}

// f16 and f8E5M2 share the exponent layout: the f16 magnitude is rounded at bit
// 8 (127 plus the kept LSB), with no exponent split and no clamp.
// CHECK-LABEL: @truncf_f16_to_f8E5M2
// CHECK-NOT: arith.truncf
// CHECK: %[[BITS:.+]] = arith.bitcast %arg0 : f16 to i16
// CHECK: %[[SRC:.+]] = arith.extui %[[BITS]] : i16 to i32
// CHECK: %[[MAG:.+]] = arith.andi %[[SRC]], %{{.+}} : i32
// CHECK-NOT: arith.minui
// CHECK: %[[C127:.+]] = arith.constant 127 : i32
// CHECK: %[[BIASED:.+]] = arith.addi %[[MAG]], %[[C127]] : i32
// CHECK: %[[UP:.+]] = arith.addi %[[BIASED]], %{{.+}} : i32
// CHECK: %[[CODE:.+]] = arith.shrui %[[UP]], %{{.+}} : i32
// CHECK-NOT: arith.minui
// CHECK: arith.ori %[[CODE]], %{{.+}} : i32
// CHECK-NOT: arith.truncf
// CHECK: %[[RESULT:.+]] = arith.bitcast %{{.+}} : i8 to f8E5M2
// CHECK: return %[[RESULT]]

// -----

func.func @truncf_vector_f32_to_f8E5M2(%arg0 : vector<4xf32>) -> vector<4xf8E5M2> {
    %0 = arith.truncf %arg0 : vector<4xf32> to vector<4xf8E5M2>
    return %0 : vector<4xf8E5M2>
}

// CHECK-LABEL: @truncf_vector_f32_to_f8E5M2
// CHECK: arith.bitcast %arg0 : vector<4xf32> to vector<4xi32>
// CHECK-NOT: arith.truncf
// CHECK: arith.bitcast %{{.+}} : vector<4xi8> to vector<4xf8E5M2>
// CHECK: return

// -----

func.func @extf_f8E4M3FN_to_f32(%arg0 : f8E4M3FN) -> f32 {
    %0 = arith.extf %arg0 : f8E4M3FN to f32
    return %0 : f32
}

// CHECK-LABEL: @extf_f8E4M3FN_to_f32
// CHECK: %[[BITS:.+]] = arith.bitcast %arg0 : f8E4M3FN to i8
// CHECK: %[[EXT:.+]] = arith.extui %[[BITS]] : i8 to i32
// CHECK-NOT: arith.extf
// The NaN 0x7F / 0xFF keeps its sign and payload under the quiet NaN.
// CHECK: %[[CNAN:.+]] = arith.constant 2143289344 : i32
// CHECK-NOT: arith.extf
// CHECK: %[[RESULT:.+]] = arith.bitcast %{{.+}} : i32 to f32
// CHECK: return %[[RESULT]]

// -----

func.func @extf_f8E4M3FN_to_f16(%arg0 : f8E4M3FN) -> f16 {
    %0 = arith.extf %arg0 : f8E4M3FN to f16
    return %0 : f16
}

// CHECK-LABEL: @extf_f8E4M3FN_to_f16
// CHECK: %[[BITS:.+]] = arith.bitcast %arg0 : f8E4M3FN to i8
// CHECK: arith.extui %[[BITS]] : i8 to i32
// CHECK-NOT: arith.truncf
// CHECK: %[[HALF:.+]] = arith.trunci %{{.+}} : i32 to i16
// CHECK: %[[RESULT:.+]] = arith.bitcast %[[HALF]] : i16 to f16
// CHECK: return %[[RESULT]]

// -----

func.func @extf_vector_f8E4M3FN_to_f32(%arg0 : vector<4xf8E4M3FN>) -> vector<4xf32> {
    %0 = arith.extf %arg0 : vector<4xf8E4M3FN> to vector<4xf32>
    return %0 : vector<4xf32>
}

// CHECK-LABEL: @extf_vector_f8E4M3FN_to_f32
// CHECK: arith.bitcast %arg0 : vector<4xf8E4M3FN> to vector<4xi8>
// CHECK-NOT: arith.extf
// CHECK: arith.bitcast %{{.+}} : vector<4xi32> to vector<4xf32>
// CHECK: return

// -----

func.func @truncf_f32_to_f8E4M3FN(%arg0 : f32) -> f8E4M3FN {
    %0 = arith.truncf %arg0 : f32 to f8E4M3FN
    return %0 : f8E4M3FN
}

// CHECK-LABEL: @truncf_f32_to_f8E4M3FN
// CHECK: %[[BITS:.+]] = arith.bitcast %arg0 : f32 to i32
// CHECK-NOT: arith.truncf
// F8E4M3FN has no infinity, so an overflowing/infinite magnitude maps to NaN:
// a code above the max-finite 126 (448.0) is replaced by the NaN 127.
// CHECK: arith.constant 126 : i32
// CHECK-NOT: arith.truncf
// CHECK: %[[CODE:.+]] = arith.trunci %{{.+}} : i32 to i8
// CHECK: %[[RESULT:.+]] = arith.bitcast %[[CODE]] : i8 to f8E4M3FN
// CHECK: return %[[RESULT]]

// -----

func.func @truncf_vector_f32_to_f8E4M3FN(%arg0 : vector<4xf32>) -> vector<4xf8E4M3FN> {
    %0 = arith.truncf %arg0 : vector<4xf32> to vector<4xf8E4M3FN>
    return %0 : vector<4xf8E4M3FN>
}

// CHECK-LABEL: @truncf_vector_f32_to_f8E4M3FN
// CHECK: arith.constant dense<126> : vector<4xi32>
// CHECK: arith.bitcast %{{.+}} : vector<4xi8> to vector<4xf8E4M3FN>
// CHECK: return
