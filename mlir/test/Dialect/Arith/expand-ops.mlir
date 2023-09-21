// RUN: mlir-opt %s -arith-expand="include-bf16=true" -split-input-file | FileCheck %s

// Test ceil divide with signed integer
// CHECK-LABEL:       func @ceildivi
// CHECK-SAME:     ([[ARG0:%.+]]: i32, [[ARG1:%.+]]: i32) -> i32 {
func.func @ceildivi(%arg0: i32, %arg1: i32) -> (i32) {
  %res = arith.ceildivsi %arg0, %arg1 : i32
  return %res : i32

// CHECK:           [[ONE:%.+]] = arith.constant 1 : i32
// CHECK:           [[ZERO:%.+]] = arith.constant 0 : i32
// CHECK:           [[MINONE:%.+]] = arith.constant -1 : i32
// CHECK:           [[CMP1:%.+]] = arith.cmpi sgt, [[ARG1]], [[ZERO]] : i32
// CHECK:           [[X:%.+]] = arith.select [[CMP1]], [[MINONE]], [[ONE]] : i32
// CHECK:           [[TRUE1:%.+]] = arith.addi [[X]], [[ARG0]] : i32
// CHECK:           [[TRUE2:%.+]] = arith.divsi [[TRUE1]], [[ARG1]] : i32
// CHECK:           [[TRUE3:%.+]] = arith.addi [[ONE]], [[TRUE2]] : i32
// CHECK:           [[FALSE1:%.+]] = arith.subi [[ZERO]], [[ARG0]] : i32
// CHECK:           [[FALSE2:%.+]] = arith.divsi [[FALSE1]], [[ARG1]] : i32
// CHECK:           [[FALSE3:%.+]] = arith.subi [[ZERO]], [[FALSE2]] : i32
// CHECK:           [[NNEG:%.+]] = arith.cmpi slt, [[ARG0]], [[ZERO]] : i32
// CHECK:           [[NPOS:%.+]] = arith.cmpi sgt, [[ARG0]], [[ZERO]] : i32
// CHECK:           [[MNEG:%.+]] = arith.cmpi slt, [[ARG1]], [[ZERO]] : i32
// CHECK:           [[MPOS:%.+]] = arith.cmpi sgt, [[ARG1]], [[ZERO]] : i32
// CHECK:           [[TERM1:%.+]] = arith.andi [[NNEG]], [[MNEG]] : i1
// CHECK:           [[TERM2:%.+]] = arith.andi [[NPOS]], [[MPOS]] : i1
// CHECK:           [[CMP2:%.+]] = arith.ori [[TERM1]], [[TERM2]] : i1
// CHECK:           [[RES:%.+]] = arith.select [[CMP2]], [[TRUE3]], [[FALSE3]] : i32
}

// -----

// Test ceil divide with index type
// CHECK-LABEL:       func @ceildivi_index
// CHECK-SAME:     ([[ARG0:%.+]]: index, [[ARG1:%.+]]: index) -> index {
func.func @ceildivi_index(%arg0: index, %arg1: index) -> (index) {
  %res = arith.ceildivsi %arg0, %arg1 : index
  return %res : index

// CHECK:           [[ONE:%.+]] = arith.constant 1 : index
// CHECK:           [[ZERO:%.+]] = arith.constant 0 : index
// CHECK:           [[MINONE:%.+]] = arith.constant -1 : index
// CHECK:           [[CMP1:%.+]] = arith.cmpi sgt, [[ARG1]], [[ZERO]] : index
// CHECK:           [[X:%.+]] = arith.select [[CMP1]], [[MINONE]], [[ONE]] : index
// CHECK:           [[TRUE1:%.+]] = arith.addi [[X]], [[ARG0]] : index
// CHECK:           [[TRUE2:%.+]] = arith.divsi [[TRUE1]], [[ARG1]] : index
// CHECK:           [[TRUE3:%.+]] = arith.addi [[ONE]], [[TRUE2]] : index
// CHECK:           [[FALSE1:%.+]] = arith.subi [[ZERO]], [[ARG0]] : index
// CHECK:           [[FALSE2:%.+]] = arith.divsi [[FALSE1]], [[ARG1]] : index
// CHECK:           [[FALSE3:%.+]] = arith.subi [[ZERO]], [[FALSE2]] : index
// CHECK:           [[NNEG:%.+]] = arith.cmpi slt, [[ARG0]], [[ZERO]] : index
// CHECK:           [[NPOS:%.+]] = arith.cmpi sgt, [[ARG0]], [[ZERO]] : index
// CHECK:           [[MNEG:%.+]] = arith.cmpi slt, [[ARG1]], [[ZERO]] : index
// CHECK:           [[MPOS:%.+]] = arith.cmpi sgt, [[ARG1]], [[ZERO]] : index
// CHECK:           [[TERM1:%.+]] = arith.andi [[NNEG]], [[MNEG]] : i1
// CHECK:           [[TERM2:%.+]] = arith.andi [[NPOS]], [[MPOS]] : i1
// CHECK:           [[CMP2:%.+]] = arith.ori [[TERM1]], [[TERM2]] : i1
// CHECK:           [[RES:%.+]] = arith.select [[CMP2]], [[TRUE3]], [[FALSE3]] : index
}

// -----

// Test floor divide with signed integer
// CHECK-LABEL:       func @floordivi
// CHECK-SAME:     ([[ARG0:%.+]]: i32, [[ARG1:%.+]]: i32) -> i32 {
func.func @floordivi(%arg0: i32, %arg1: i32) -> (i32) {
  %res = arith.floordivsi %arg0, %arg1 : i32
  return %res : i32
// CHECK:           [[ONE:%.+]] = arith.constant 1 : i32
// CHECK:           [[ZERO:%.+]] = arith.constant 0 : i32
// CHECK:           [[MIN1:%.+]] = arith.constant -1 : i32
// CHECK:           [[CMP1:%.+]] = arith.cmpi slt, [[ARG1]], [[ZERO]] : i32
// CHECK:           [[X:%.+]] = arith.select [[CMP1]], [[ONE]], [[MIN1]] : i32
// CHECK:           [[TRUE1:%.+]] = arith.subi [[X]], [[ARG0]] : i32
// CHECK:           [[TRUE2:%.+]] = arith.divsi [[TRUE1]], [[ARG1]] : i32
// CHECK:           [[TRUE3:%.+]] = arith.subi [[MIN1]], [[TRUE2]] : i32
// CHECK:           [[FALSE:%.+]] = arith.divsi [[ARG0]], [[ARG1]] : i32
// CHECK:           [[NNEG:%.+]] = arith.cmpi slt, [[ARG0]], [[ZERO]] : i32
// CHECK:           [[NPOS:%.+]] = arith.cmpi sgt, [[ARG0]], [[ZERO]] : i32
// CHECK:           [[MNEG:%.+]] = arith.cmpi slt, [[ARG1]], [[ZERO]] : i32
// CHECK:           [[MPOS:%.+]] = arith.cmpi sgt, [[ARG1]], [[ZERO]] : i32
// CHECK:           [[TERM1:%.+]] = arith.andi [[NNEG]], [[MPOS]] : i1
// CHECK:           [[TERM2:%.+]] = arith.andi [[NPOS]], [[MNEG]] : i1
// CHECK:           [[CMP2:%.+]] = arith.ori [[TERM1]], [[TERM2]] : i1
// CHECK:           [[RES:%.+]] = arith.select [[CMP2]], [[TRUE3]], [[FALSE]] : i32
}

// -----

// Test floor divide with index type
// CHECK-LABEL:       func @floordivi_index
// CHECK-SAME:     ([[ARG0:%.+]]: index, [[ARG1:%.+]]: index) -> index {
func.func @floordivi_index(%arg0: index, %arg1: index) -> (index) {
  %res = arith.floordivsi %arg0, %arg1 : index
  return %res : index
// CHECK:           [[ONE:%.+]] = arith.constant 1 : index
// CHECK:           [[ZERO:%.+]] = arith.constant 0 : index
// CHECK:           [[MIN1:%.+]] = arith.constant -1 : index
// CHECK:           [[CMP1:%.+]] = arith.cmpi slt, [[ARG1]], [[ZERO]] : index
// CHECK:           [[X:%.+]] = arith.select [[CMP1]], [[ONE]], [[MIN1]] : index
// CHECK:           [[TRUE1:%.+]] = arith.subi [[X]], [[ARG0]] : index
// CHECK:           [[TRUE2:%.+]] = arith.divsi [[TRUE1]], [[ARG1]] : index
// CHECK:           [[TRUE3:%.+]] = arith.subi [[MIN1]], [[TRUE2]] : index
// CHECK:           [[FALSE:%.+]] = arith.divsi [[ARG0]], [[ARG1]] : index
// CHECK:           [[NNEG:%.+]] = arith.cmpi slt, [[ARG0]], [[ZERO]] : index
// CHECK:           [[NPOS:%.+]] = arith.cmpi sgt, [[ARG0]], [[ZERO]] : index
// CHECK:           [[MNEG:%.+]] = arith.cmpi slt, [[ARG1]], [[ZERO]] : index
// CHECK:           [[MPOS:%.+]] = arith.cmpi sgt, [[ARG1]], [[ZERO]] : index
// CHECK:           [[TERM1:%.+]] = arith.andi [[NNEG]], [[MPOS]] : i1
// CHECK:           [[TERM2:%.+]] = arith.andi [[NPOS]], [[MNEG]] : i1
// CHECK:           [[CMP2:%.+]] = arith.ori [[TERM1]], [[TERM2]] : i1
// CHECK:           [[RES:%.+]] = arith.select [[CMP2]], [[TRUE3]], [[FALSE]] : index
}

// -----

// Test floor divide with vector
// CHECK-LABEL:   func.func @floordivi_vec(
// CHECK-SAME:                             %[[VAL_0:.*]]: vector<4xi32>,
// CHECK-SAME:                             %[[VAL_1:.*]]: vector<4xi32>) -> vector<4xi32> {
func.func @floordivi_vec(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> (vector<4xi32>) {
  %res = arith.floordivsi %arg0, %arg1 : vector<4xi32>
  return %res : vector<4xi32>
// CHECK:           %[[VAL_2:.*]] = arith.constant dense<1> : vector<4xi32>
// CHECK:           %[[VAL_3:.*]] = arith.constant dense<0> : vector<4xi32>
// CHECK:           %[[VAL_4:.*]] = arith.constant dense<-1> : vector<4xi32>
// CHECK:           %[[VAL_5:.*]] = arith.cmpi slt, %[[VAL_1]], %[[VAL_3]] : vector<4xi32>
// CHECK:           %[[VAL_6:.*]] = arith.select %[[VAL_5]], %[[VAL_2]], %[[VAL_4]] : vector<4xi1>, vector<4xi32>
// CHECK:           %[[VAL_7:.*]] = arith.subi %[[VAL_6]], %[[VAL_0]] : vector<4xi32>
// CHECK:           %[[VAL_8:.*]] = arith.divsi %[[VAL_7]], %[[VAL_1]] : vector<4xi32>
// CHECK:           %[[VAL_9:.*]] = arith.subi %[[VAL_4]], %[[VAL_8]] : vector<4xi32>
// CHECK:           %[[VAL_10:.*]] = arith.divsi %[[VAL_0]], %[[VAL_1]] : vector<4xi32>
// CHECK:           %[[VAL_11:.*]] = arith.cmpi slt, %[[VAL_0]], %[[VAL_3]] : vector<4xi32>
// CHECK:           %[[VAL_12:.*]] = arith.cmpi sgt, %[[VAL_0]], %[[VAL_3]] : vector<4xi32>
// CHECK:           %[[VAL_13:.*]] = arith.cmpi slt, %[[VAL_1]], %[[VAL_3]] : vector<4xi32>
// CHECK:           %[[VAL_14:.*]] = arith.cmpi sgt, %[[VAL_1]], %[[VAL_3]] : vector<4xi32>
// CHECK:           %[[VAL_15:.*]] = arith.andi %[[VAL_11]], %[[VAL_14]] : vector<4xi1>
// CHECK:           %[[VAL_16:.*]] = arith.andi %[[VAL_12]], %[[VAL_13]] : vector<4xi1>
// CHECK:           %[[VAL_17:.*]] = arith.ori %[[VAL_15]], %[[VAL_16]] : vector<4xi1>
// CHECK:           %[[VAL_18:.*]] = arith.select %[[VAL_17]], %[[VAL_9]], %[[VAL_10]] : vector<4xi1>, vector<4xi32>
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

// CHECK-LABEL: func @maxf
func.func @maxf(%a: f32, %b: f32) -> f32 {
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

// CHECK-LABEL: func @maxf_vector
func.func @maxf_vector(%a: vector<4xf16>, %b: vector<4xf16>) -> vector<4xf16> {
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

// CHECK-LABEL: func @minf
func.func @minf(%a: f32, %b: f32) -> f32 {
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

func.func @truncf_f32(%arg0 : f32) -> bf16 {
    %0 = arith.truncf %arg0 : f32 to bf16
    return %0 : bf16
}

// CHECK-LABEL: @truncf_f32

// CHECK-DAG: %[[C16:.+]] = arith.constant 16
// CHECK-DAG: %[[C32768:.+]] = arith.constant 32768
// CHECK-DAG: %[[C2130706432:.+]] = arith.constant 2130706432
// CHECK-DAG: %[[C2139095040:.+]] = arith.constant 2139095040
// CHECK-DAG: %[[C8388607:.+]] = arith.constant 8388607
// CHECK-DAG: %[[C31:.+]] = arith.constant 31
// CHECK-DAG: %[[C23:.+]] = arith.constant 23
// CHECK-DAG: %[[BITCAST:.+]] = arith.bitcast %arg0
// CHECK-DAG: %[[SIGN:.+]] = arith.shrui %[[BITCAST:.+]], %[[C31]]
// CHECK-DAG: %[[ROUND:.+]] = arith.subi %[[C32768]], %[[SIGN]]
// CHECK-DAG: %[[MANTISSA:.+]] = arith.andi %[[BITCAST]], %[[C8388607]]
// CHECK-DAG: %[[ROUNDED:.+]] = arith.addi %[[MANTISSA]], %[[ROUND]]
// CHECK-DAG: %[[ROLL:.+]] = arith.shrui %[[ROUNDED]], %[[C23]]
// CHECK-DAG: %[[SHR:.+]] = arith.shrui %[[ROUNDED]], %[[ROLL]]
// CHECK-DAG: %[[EXP:.+]] = arith.andi %0, %[[C2139095040]]
// CHECK-DAG: %[[EXPROUND:.+]] = arith.addi %[[EXP]], %[[ROUNDED]]
// CHECK-DAG: %[[EXPROLL:.+]] = arith.andi %[[EXPROUND]], %[[C2139095040]]
// CHECK-DAG: %[[EXPMAX:.+]] = arith.cmpi uge, %[[EXP]], %[[C2130706432]]
// CHECK-DAG: %[[EXPNEW:.+]] = arith.select %[[EXPMAX]], %[[EXP]], %[[EXPROLL]]
// CHECK-DAG: %[[OVERFLOW_B:.+]] = arith.trunci %[[ROLL]]
// CHECK-DAG: %[[KEEP_MAN:.+]] = arith.andi %[[EXPMAX]], %[[OVERFLOW_B]]
// CHECK-DAG: %[[MANNEW:.+]] = arith.select %[[KEEP_MAN]], %[[MANTISSA]], %[[SHR]]
// CHECK-DAG: %[[NEWSIGN:.+]] = arith.shli %[[SIGN]], %[[C31]]
// CHECK-DAG: %[[WITHEXP:.+]] = arith.ori %[[NEWSIGN]], %[[EXPNEW]]
// CHECK-DAG: %[[WITHMAN:.+]] = arith.ori %[[WITHEXP]], %[[MANNEW]]
// CHECK-DAG: %[[SHIFT:.+]] = arith.shrui %[[WITHMAN]], %[[C16]]
// CHECK-DAG: %[[TRUNC:.+]] = arith.trunci %[[SHIFT]]
// CHECK-DAG: %[[RES:.+]] = arith.bitcast %[[TRUNC]]
// CHECK: return %[[RES]]

// -----

func.func @truncf_vector_f32(%arg0 : vector<4xf32>) -> vector<4xbf16> {
    %0 = arith.truncf %arg0 : vector<4xf32> to vector<4xbf16>
    return %0 : vector<4xbf16>
}

// CHECK-LABEL: @truncf_vector_f32
// CHECK-NOT: arith.truncf
