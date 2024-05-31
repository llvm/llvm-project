// RUN: mlir-opt -split-input-file -convert-arith-to-emitc %s | FileCheck %s

// CHECK-LABEL: arith_constants
func.func @arith_constants() {
  // CHECK: emitc.constant
  // CHECK-SAME: value = 0 : index
  %c_index = arith.constant 0 : index
  // CHECK: emitc.constant
  // CHECK-SAME: value = 0 : i32
  %c_signless_int_32 = arith.constant 0 : i32
  // CHECK: emitc.constant
  // CHECK-SAME: value = 0.{{0+}}e+00 : f32
  %c_float_32 = arith.constant 0.0 : f32
  // CHECK: emitc.constant
  // CHECK-SAME: value = dense<0> : tensor<i32>
  %c_tensor_single_value = arith.constant dense<0> : tensor<i32>
  // CHECK: emitc.constant
  // CHECK-SAME: value{{.*}}[1, 2], [-3, 9], [0, 0], [2, -1]{{.*}}tensor<4x2xi64>
  %c_tensor_value = arith.constant dense<[[1, 2], [-3, 9], [0, 0], [2, -1]]> : tensor<4x2xi64>
  return
}

// -----

func.func @arith_ops(%arg0: f32, %arg1: f32) {
  // CHECK: [[V0:[^ ]*]] = emitc.add %arg0, %arg1 : (f32, f32) -> f32
  %0 = arith.addf %arg0, %arg1 : f32
  // CHECK: [[V1:[^ ]*]] = emitc.div %arg0, %arg1 : (f32, f32) -> f32
  %1 = arith.divf %arg0, %arg1 : f32  
  // CHECK: [[V2:[^ ]*]] = emitc.mul %arg0, %arg1 : (f32, f32) -> f32
  %2 = arith.mulf %arg0, %arg1 : f32
  // CHECK: [[V3:[^ ]*]] = emitc.sub %arg0, %arg1 : (f32, f32) -> f32
  %3 = arith.subf %arg0, %arg1 : f32

  return
}

// -----

// CHECK-LABEL: arith_integer_ops
func.func @arith_integer_ops(%arg0: i32, %arg1: i32) {
  // CHECK: %[[C1:[^ ]*]] = emitc.cast %arg0 : i32 to ui32
  // CHECK: %[[C2:[^ ]*]] = emitc.cast %arg1 : i32 to ui32
  // CHECK: %[[ADD:[^ ]*]] = emitc.add %[[C1]], %[[C2]] : (ui32, ui32) -> ui32
  // CHECK: %[[C3:[^ ]*]] = emitc.cast %[[ADD]] : ui32 to i32
  %0 = arith.addi %arg0, %arg1 : i32
  // CHECK: %[[C1:[^ ]*]] = emitc.cast %arg0 : i32 to ui32
  // CHECK: %[[C2:[^ ]*]] = emitc.cast %arg1 : i32 to ui32
  // CHECK: %[[SUB:[^ ]*]] = emitc.sub %[[C1]], %[[C2]] : (ui32, ui32) -> ui32
  // CHECK: %[[C3:[^ ]*]] = emitc.cast %[[SUB]] : ui32 to i32
  %1 = arith.subi %arg0, %arg1 : i32
  // CHECK: %[[C1:[^ ]*]] = emitc.cast %arg0 : i32 to ui32
  // CHECK: %[[C2:[^ ]*]] = emitc.cast %arg1 : i32 to ui32
  // CHECK: %[[MUL:[^ ]*]] = emitc.mul %[[C1]], %[[C2]] : (ui32, ui32) -> ui32
  // CHECK: %[[C3:[^ ]*]] = emitc.cast %[[MUL]] : ui32 to i32
  %2 = arith.muli %arg0, %arg1 : i32

  return
}

// -----

// CHECK-LABEL: arith_integer_ops_signed_nsw
func.func @arith_integer_ops_signed_nsw(%arg0: i32, %arg1: i32) {
  // CHECK: emitc.add %arg0, %arg1 : (i32, i32) -> i32
  %0 = arith.addi %arg0, %arg1 overflow<nsw> : i32
  // CHECK: emitc.sub %arg0, %arg1 : (i32, i32) -> i32
  %1 = arith.subi %arg0, %arg1 overflow<nsw>  : i32
  // CHECK: emitc.mul %arg0, %arg1 : (i32, i32) -> i32
  %2 = arith.muli %arg0, %arg1 overflow<nsw> : i32

  return
}

// -----

// CHECK-LABEL: arith_index
func.func @arith_index(%arg0: index, %arg1: index) {
  // CHECK: emitc.add %arg0, %arg1 : (index, index) -> index
  %0 = arith.addi %arg0, %arg1 : index
  // CHECK: emitc.sub %arg0, %arg1 : (index, index) -> index
  %1 = arith.subi %arg0, %arg1 : index
  // CHECK: emitc.mul %arg0, %arg1 : (index, index) -> index
  %2 = arith.muli %arg0, %arg1 : index

  return
}

// -----

// CHECK-LABEL: arith_signed_integer_div_rem
func.func @arith_signed_integer_div_rem(%arg0: i32, %arg1: i32) {
  // CHECK: emitc.div %arg0, %arg1 : (i32, i32) -> i32
  %0 = arith.divsi %arg0, %arg1 : i32
  // CHECK: emitc.rem %arg0, %arg1 : (i32, i32) -> i32
  %1 = arith.remsi %arg0, %arg1 : i32
  return
}

// -----

func.func @arith_select(%arg0: i1, %arg1: tensor<8xi32>, %arg2: tensor<8xi32>) -> () {
  // CHECK: [[V0:[^ ]*]] = emitc.conditional %arg0, %arg1, %arg2 : tensor<8xi32>
  %0 = arith.select %arg0, %arg1, %arg2 : i1, tensor<8xi32>
  return
}

// -----

func.func @arith_cmpi_eq(%arg0: i32, %arg1: i32) -> i1 {
  // CHECK-LABEL: arith_cmpi_eq
  // CHECK-SAME: ([[Arg0:[^ ]*]]: i32, [[Arg1:[^ ]*]]: i32)
  // CHECK-DAG: [[EQ:[^ ]*]] = emitc.cmp eq, [[Arg0]], [[Arg1]] : (i32, i32) -> i1
  %eq = arith.cmpi eq, %arg0, %arg1 : i32
  // CHECK: return [[EQ]]
  return %eq: i1
}

func.func @arith_cmpi_ult(%arg0: i32, %arg1: i32) -> i1 {
  // CHECK-LABEL: arith_cmpi_ult
  // CHECK-SAME: ([[Arg0:[^ ]*]]: i32, [[Arg1:[^ ]*]]: i32)
  // CHECK-DAG: [[CastArg0:[^ ]*]] = emitc.cast [[Arg0]] : i32 to ui32
  // CHECK-DAG: [[CastArg1:[^ ]*]] = emitc.cast [[Arg1]] : i32 to ui32
  // CHECK-DAG: [[ULT:[^ ]*]] = emitc.cmp lt, [[CastArg0]], [[CastArg1]] : (ui32, ui32) -> i1
  %ult = arith.cmpi ult, %arg0, %arg1 : i32

  // CHECK: return [[ULT]]
  return %ult: i1
}

func.func @arith_cmpi_predicates(%arg0: i32, %arg1: i32) {
  // CHECK: emitc.cmp lt, {{.*}} : (ui32, ui32) -> i1
  %ult = arith.cmpi ult, %arg0, %arg1 : i32
  // CHECK: emitc.cmp lt, {{.*}} : (i32, i32) -> i1
  %slt = arith.cmpi slt, %arg0, %arg1 : i32
  // CHECK: emitc.cmp le, {{.*}} : (ui32, ui32) -> i1
  %ule = arith.cmpi ule, %arg0, %arg1 : i32
  // CHECK: emitc.cmp le, {{.*}} : (i32, i32) -> i1
  %sle = arith.cmpi sle, %arg0, %arg1 : i32
  // CHECK: emitc.cmp gt, {{.*}} : (ui32, ui32) -> i1
  %ugt = arith.cmpi ugt, %arg0, %arg1 : i32
  // CHECK: emitc.cmp gt, {{.*}} : (i32, i32) -> i1
  %sgt = arith.cmpi sgt, %arg0, %arg1 : i32
  // CHECK: emitc.cmp ge, {{.*}} : (ui32, ui32) -> i1
  %uge = arith.cmpi uge, %arg0, %arg1 : i32
  // CHECK: emitc.cmp ge, {{.*}} : (i32, i32) -> i1
  %sge = arith.cmpi sge, %arg0, %arg1 : i32
  // CHECK: emitc.cmp eq, {{.*}} : (i32, i32) -> i1
  %eq = arith.cmpi eq, %arg0, %arg1 : i32
  // CHECK: emitc.cmp ne, {{.*}} : (i32, i32) -> i1
  %ne = arith.cmpi ne, %arg0, %arg1 : i32
  
  return
}

// -----

func.func @arith_float_to_int_cast_ops(%arg0: f32, %arg1: f64) {
  // CHECK: emitc.cast %arg0 : f32 to i32
  %0 = arith.fptosi %arg0 : f32 to i32

  // CHECK: emitc.cast %arg1 : f64 to i32
  %1 = arith.fptosi %arg1 : f64 to i32

  // CHECK: emitc.cast %arg0 : f32 to i16
  %2 = arith.fptosi %arg0 : f32 to i16

  // CHECK: emitc.cast %arg1 : f64 to i16
  %3 = arith.fptosi %arg1 : f64 to i16

  // CHECK: %[[CAST0:.*]] = emitc.cast %arg0 : f32 to ui32
  // CHECK: emitc.cast %[[CAST0]] : ui32 to i32
  %4 = arith.fptoui %arg0 : f32 to i32

  return
}

func.func @arith_int_to_float_cast_ops(%arg0: i8, %arg1: i64) {
  // CHECK: emitc.cast %arg0 : i8 to f32
  %0 = arith.sitofp %arg0 : i8 to f32

  // CHECK: emitc.cast %arg1 : i64 to f32
  %1 = arith.sitofp %arg1 : i64 to f32

  // CHECK: %[[CAST_UNS:.*]] = emitc.cast %arg0 : i8 to ui8
  // CHECK: emitc.cast %[[CAST_UNS]] : ui8 to f32
  %2 = arith.uitofp %arg0 : i8 to f32

  return
}

// -----

func.func @arith_trunci(%arg0: i32) -> i8 {
  // CHECK-LABEL: arith_trunci
  // CHECK-SAME: (%[[Arg0:[^ ]*]]: i32)
  // CHECK: %[[CastUI:.*]] = emitc.cast %[[Arg0]] : i32 to ui32
  // CHECK: %[[Trunc:.*]] = emitc.cast %[[CastUI]] : ui32 to ui8
  // CHECK: emitc.cast %[[Trunc]] : ui8 to i8
  %truncd = arith.trunci %arg0 : i32 to i8

  return %truncd : i8
}

// -----

func.func @arith_trunci_to_i1(%arg0: i32) -> i1 {
  // CHECK-LABEL: arith_trunci_to_i1
  // CHECK-SAME: (%[[Arg0:[^ ]*]]: i32)
  // CHECK: %[[Const:.*]] = "emitc.constant"
  // CHECK-SAME: value = 1
  // CHECK: %[[And:.*]] = emitc.bitwise_and %[[Arg0]], %[[Const]] : (i32, i32) -> i32
  // CHECK: emitc.cast %[[And]] : i32 to i1
  %truncd = arith.trunci %arg0 : i32 to i1

  return %truncd : i1
}

// -----

func.func @arith_extsi(%arg0: i32) {
  // CHECK-LABEL: arith_extsi
  // CHECK-SAME: ([[Arg0:[^ ]*]]: i32)
  // CHECK: emitc.cast [[Arg0]] : i32 to i64
  %extd = arith.extsi %arg0 : i32 to i64

  return
}

// -----

func.func @arith_extui(%arg0: i32) {
  // CHECK-LABEL: arith_extui
  // CHECK-SAME: (%[[Arg0:[^ ]*]]: i32)
  // CHECK: %[[Conv0:.*]] = emitc.cast %[[Arg0]] : i32 to ui32
  // CHECK: %[[Conv1:.*]] = emitc.cast %[[Conv0]] : ui32 to ui64
  // CHECK: emitc.cast %[[Conv1]] : ui64 to i64
  %extd = arith.extui %arg0 : i32 to i64

  return
}

// -----

func.func @arith_extui_i1_to_i32(%arg0: i1) {
  // CHECK-LABEL: arith_extui_i1_to_i32
  // CHECK-SAME: (%[[Arg0:[^ ]*]]: i1)
  // CHECK: %[[Conv0:.*]] = emitc.cast %[[Arg0]] : i1 to ui1
  // CHECK: %[[Conv1:.*]] = emitc.cast %[[Conv0]] : ui1 to ui32
  // CHECK: emitc.cast %[[Conv1]] : ui32 to i32
  %idx = arith.extui %arg0 : i1 to i32
  return
}
