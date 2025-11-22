// RUN: mlir-opt -split-input-file -convert-arith-to-emitc %s | FileCheck %s
// RUN: mlir-opt -split-input-file -convert-to-emitc="filter-dialects=arith" %s | FileCheck %s

// CHECK-LABEL: arith_constants
func.func @arith_constants() {
  // CHECK: emitc.constant
  // CHECK-SAME: value = 0
  // CHECK-SAME: () -> !emitc.size_t
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
func.func @arith_index(%arg0: i32, %arg1: i32) {
  // CHECK: %[[CST0:.*]] = emitc.cast %{{.*}} : {{.*}} to !emitc.size_t
  %cst0 = arith.index_cast %arg0 : i32 to index
  // CHECK: %[[CST1:.*]] = emitc.cast %{{.*}} : {{.*}} to !emitc.size_t
  %cst1 = arith.index_cast %arg1 : i32 to index

  // CHECK: emitc.add %[[CST0]], %[[CST1]] : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
  %0 = arith.addi %cst0, %cst1 : index
  // CHECK: emitc.sub %[[CST0]], %[[CST1]] : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
  %1 = arith.subi %cst0, %cst1 : index
  // CHECK: emitc.mul %[[CST0]], %[[CST1]] : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
  %2 = arith.muli %cst0, %cst1 : index

  return
}

// -----

// CHECK-LABEL: arith_bitwise
// CHECK-SAME: %[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32
func.func @arith_bitwise(%arg0: i32, %arg1: i32) {
  // CHECK: %[[C1:[^ ]*]] = emitc.cast %[[ARG0]] : i32 to ui32
  // CHECK: %[[C2:[^ ]*]] = emitc.cast %[[ARG1]] : i32 to ui32
  // CHECK: %[[AND:[^ ]*]] = emitc.bitwise_and %[[C1]], %[[C2]] : (ui32, ui32) -> ui32
  // CHECK: %[[C3:[^ ]*]] = emitc.cast %[[AND]] : ui32 to i32
  %0 = arith.andi %arg0, %arg1 : i32
  // CHECK: %[[C1:[^ ]*]] = emitc.cast %[[ARG0]] : i32 to ui32
  // CHECK: %[[C2:[^ ]*]] = emitc.cast %[[ARG1]] : i32 to ui32
  // CHECK: %[[OR:[^ ]*]] = emitc.bitwise_or %[[C1]], %[[C2]] : (ui32, ui32) -> ui32
  // CHECK: %[[C3:[^ ]*]] = emitc.cast %[[OR]] : ui32 to i32
  %1 = arith.ori %arg0, %arg1 : i32
  // CHECK: %[[C1:[^ ]*]] = emitc.cast %[[ARG0]] : i32 to ui32
  // CHECK: %[[C2:[^ ]*]] = emitc.cast %[[ARG1]] : i32 to ui32
  // CHECK: %[[XOR:[^ ]*]] = emitc.bitwise_xor %[[C1]], %[[C2]] : (ui32, ui32) -> ui32
  // CHECK: %[[C3:[^ ]*]] = emitc.cast %[[XOR]] : ui32 to i32
  %2 = arith.xori %arg0, %arg1 : i32

  return
}

// -----

// CHECK-LABEL: arith_bitwise_bool
// CHECK-SAME: %[[ARG0:.*]]: i1, %[[ARG1:.*]]: i1
func.func @arith_bitwise_bool(%arg0: i1, %arg1: i1) {
  // CHECK: %[[AND:[^ ]*]] = emitc.bitwise_and %[[ARG0]], %[[ARG1]] : (i1, i1) -> i1
  %0 = arith.andi %arg0, %arg1 : i1
  // CHECK: %[[OR:[^ ]*]] = emitc.bitwise_or %[[ARG0]], %[[ARG1]] : (i1, i1) -> i1
  %1 = arith.ori %arg0, %arg1 : i1
  // CHECK: %[[xor:[^ ]*]] = emitc.bitwise_xor %[[ARG0]], %[[ARG1]] : (i1, i1) -> i1
  %2 = arith.xori %arg0, %arg1 : i1
  
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

// CHECK-LABEL: arith_shift_left
// CHECK-SAME: %[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32
func.func @arith_shift_left(%arg0: i32, %arg1: i32) {
  // CHECK-DAG: %[[C1:[^ ]*]] = emitc.cast %[[ARG0]] : i32 to ui32
  // CHECK-DAG: %[[C2:[^ ]*]] = emitc.cast %[[ARG1]] : i32 to ui32
  // CHECK-DAG: %[[SizeConstant:[^ ]*]] = "emitc.constant"{{.*}}value = 32
  // CHECK-DAG: %[[CmpNoExcess:[^ ]*]] = emitc.cmp lt, %[[C2]], %[[SizeConstant]] : (ui32, ui32) -> i1
  // CHECK-DAG: %[[Zero:[^ ]*]] = "emitc.constant"{{.*}}value = 0
  // CHECK:      %[[ShiftRes:[^ ]*]] = emitc.expression %[[C1]], %[[C2]], %[[CmpNoExcess]], %[[Zero]] : (ui32, ui32, i1, ui32) -> ui32 {
  // CHECK-NEXT:   %[[SHL:[^ ]*]] = bitwise_left_shift %[[C1]], %[[C2]] : (ui32, ui32) -> ui32
  // CHECK-NEXT:   %[[Ternary:[^ ]*]] = conditional %[[CmpNoExcess]], %[[SHL]], %[[Zero]] : ui32
  // CHECK-NEXT:   yield %[[Ternary]] : ui32
  // CHECK-NEXT: }
  // CHECK-NEXT: emitc.cast %[[ShiftRes]] : ui32 to i32
  %1 = arith.shli %arg0, %arg1 : i32
  return
}

// -----

// CHECK-LABEL: arith_shift_right
// CHECK-SAME: %[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32
func.func @arith_shift_right(%arg0: i32, %arg1: i32) {
  // CHECK-DAG: %[[C1:[^ ]*]] = emitc.cast %[[ARG0]] : i32 to ui32
  // CHECK-DAG: %[[C2:[^ ]*]] = emitc.cast %[[ARG1]] : i32 to ui32
  // CHECK-DAG: %[[SizeConstant:[^ ]*]] = "emitc.constant"{{.*}}value = 32{{.*}}ui32
  // CHECK-DAG: %[[CmpNoExcess:[^ ]*]] = emitc.cmp lt, %[[C2]], %[[SizeConstant]] : (ui32, ui32) -> i1
  // CHECK-DAG: %[[Zero:[^ ]*]] = "emitc.constant"{{.*}}value = 0{{.*}}ui32
  // CHECK:      %[[ShiftRes:[^ ]*]] = emitc.expression %[[C1]], %[[C2]], %[[CmpNoExcess]], %[[Zero]] : (ui32, ui32, i1, ui32) -> ui32 {
  // CHECK-NEXT:   %[[SHR:[^ ]*]] = bitwise_right_shift %[[C1]], %[[C2]] : (ui32, ui32) -> ui32
  // CHECK-NEXT:   %[[Ternary:[^ ]*]] = conditional %[[CmpNoExcess]], %[[SHR]], %[[Zero]] : ui32
  // CHECK-NEXT:   yield %[[Ternary]] : ui32
  // CHECK-NEXT: }
  // CHECK-NEXT: emitc.cast %[[ShiftRes]] : ui32 to i32
  %2 = arith.shrui %arg0, %arg1 : i32

  // CHECK-DAG: %[[SC2:[^ ]*]] = emitc.cast %[[ARG1]] : i32 to ui32
  // CHECK-DAG: %[[SSizeConstant:[^ ]*]] = "emitc.constant"{{.*}}value = 32{{.*}}ui32
  // CHECK-DAG: %[[SCmpNoExcess:[^ ]*]] = emitc.cmp lt, %[[SC2]], %[[SSizeConstant]] : (ui32, ui32) -> i1
  // CHECK-DAG: %[[SZero:[^ ]*]] = "emitc.constant"{{.*}}value = 0{{.*}}i32
  // CHECK:      %[[SShiftRes:[^ ]*]] = emitc.expression %[[ARG0]], %[[SC2]], %[[SCmpNoExcess]], %[[SZero]] : (i32, ui32, i1, i32) -> i32 {
  // CHECK-NEXT:   %[[SHRSI:[^ ]*]] = bitwise_right_shift %[[ARG0]], %[[SC2]] : (i32, ui32) -> i32
  // CHECK-NEXT:   %[[STernary:[^ ]*]] = conditional %[[SCmpNoExcess]], %[[SHRSI]], %[[SZero]] : i32
  // CHECK-NEXT:   yield %[[STernary]] : i32
  // CHECK-NEXT: }
  %3 = arith.shrsi %arg0, %arg1 : i32

  return
}

// -----

// CHECK-LABEL: arith_shift_left_index
// CHECK-SAME: %[[AMOUNT:.*]]: i32
func.func @arith_shift_left_index(%amount: i32) {
  %cst0 = "arith.constant"() {value = 42 : index} : () -> (index)
  %cast1 = arith.index_cast %amount : i32 to index
  // CHECK-DAG: %[[C1:[^ ]*]] = "emitc.constant"(){{.*}}value = 42{{.*}}!emitc.size_t
  // CHECK-DAG: %[[Cast1:[^ ]*]] = emitc.cast %[[AMOUNT]] : i32 to !emitc.ptrdiff_t
  // CHECK-DAG: %[[AmountIdx:[^ ]*]] = emitc.cast %[[Cast1]] : !emitc.ptrdiff_t to !emitc.size_t
  // CHECK-DAG: %[[Byte:[^ ]*]] = "emitc.constant"{{.*}}value = 8{{.*}}index
  // CHECK-DAG: %[[SizeOf:[^ ]*]] = emitc.call_opaque "sizeof"(%[[Byte]]) : (!emitc.size_t) -> !emitc.size_t
  // CHECK-DAG: %[[SizeConstant:[^ ]*]] = emitc.mul %[[Byte]], %[[SizeOf]] : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
  // CHECK-DAG: %[[CmpNoExcess:[^ ]*]] = emitc.cmp lt, %[[AmountIdx]], %[[SizeConstant]] : (!emitc.size_t, !emitc.size_t) -> i1
  // CHECK-DAG: %[[Zero:[^ ]*]] = "emitc.constant"{{.*}}value = 0
  // CHECK:      %[[ShiftRes:[^ ]*]] = emitc.expression %[[C1]], %[[AmountIdx]], %[[CmpNoExcess]], %[[Zero]] : (!emitc.size_t, !emitc.size_t, i1, !emitc.size_t) -> !emitc.size_t {
  // CHECK-NEXT:   %[[SHL:[^ ]*]] = bitwise_left_shift %[[C1]], %[[AmountIdx]] : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
  // CHECK-NEXT:   %[[Ternary:[^ ]*]] = conditional %[[CmpNoExcess]], %[[SHL]], %[[Zero]] : !emitc.size_t
  // CHECK-NEXT:   yield %[[Ternary]] : !emitc.size_t
  // CHECK-NEXT: }
  %1 = arith.shli %cst0, %cast1 : index
  return
}

// -----

// CHECK-LABEL: arith_shift_right_index
// CHECK-SAME: %[[AMOUNT:.*]]: i32
func.func @arith_shift_right_index(%amount: i32) {
  // CHECK-DAG: %[[C1:[^ ]*]] = "emitc.constant"(){{.*}}value = 42{{.*}}!emitc.size_t
  // CHECK-DAG: %[[Cast1:[^ ]*]] = emitc.cast %[[AMOUNT]] : i32 to !emitc.ptrdiff_t
  // CHECK-DAG: %[[AmountIdx:[^ ]*]] = emitc.cast %[[Cast1]] : !emitc.ptrdiff_t to !emitc.size_t
  %arg0 = "arith.constant"() {value = 42 : index} : () -> (index)
  %arg1 = arith.index_cast %amount : i32 to index

  // CHECK-DAG: %[[Byte:[^ ]*]] = "emitc.constant"{{.*}}value = 8{{.*}}index
  // CHECK-DAG: %[[SizeOf:[^ ]*]] = emitc.call_opaque "sizeof"(%[[Byte]]) : (!emitc.size_t) -> !emitc.size_t
  // CHECK-DAG: %[[SizeConstant:[^ ]*]] = emitc.mul %[[Byte]], %[[SizeOf]] : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
  // CHECK-DAG: %[[CmpNoExcess:[^ ]*]] = emitc.cmp lt, %[[AmountIdx]], %[[SizeConstant]] : (!emitc.size_t, !emitc.size_t) -> i1
  // CHECK-DAG: %[[Zero:[^ ]*]] = "emitc.constant"{{.*}}value = 0{{.*}}!emitc.size_t
  // CHECK:      %[[ShiftRes:[^ ]*]] = emitc.expression %[[C1]], %[[AmountIdx]], %[[CmpNoExcess]], %[[Zero]] : (!emitc.size_t, !emitc.size_t, i1, !emitc.size_t) -> !emitc.size_t {
  // CHECK-NEXT:   %[[SHR:[^ ]*]] = bitwise_right_shift %[[C1]], %[[AmountIdx]] : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
  // CHECK-NEXT:   %[[Ternary:[^ ]*]] = conditional %[[CmpNoExcess]], %[[SHR]], %[[Zero]] : !emitc.size_t
  // CHECK-NEXT:   yield %[[Ternary]] : !emitc.size_t
  // CHECK-NEXT: }
  %2 = arith.shrui %arg0, %arg1 : index

  // CHECK-DAG: %[[SC1:[^ ]*]] = emitc.cast %[[C1]] : !emitc.size_t to !emitc.ptrdiff_t
  // CHECK-DAG: %[[SByte:[^ ]*]] = "emitc.constant"{{.*}}value = 8{{.*}}index{{.*}}!emitc.size_t
  // CHECK-DAG: %[[SSizeOf:[^ ]*]] = emitc.call_opaque "sizeof"(%[[SByte]]) : (!emitc.size_t) -> !emitc.size_t
  // CHECK-DAG: %[[SSizeConstant:[^ ]*]] = emitc.mul %[[SByte]], %[[SSizeOf]] : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
  // CHECK-DAG: %[[SCmpNoExcess:[^ ]*]] = emitc.cmp lt, %[[AmountIdx]], %[[SSizeConstant]] : (!emitc.size_t, !emitc.size_t) -> i1
  // CHECK-DAG: %[[SZero:[^ ]*]] = "emitc.constant"{{.*}}value = 0{{.*}}!emitc.ptrdiff_t
  // CHECK:      %[[SShiftRes:[^ ]*]] = emitc.expression %[[SC1]], %[[AmountIdx]], %[[SCmpNoExcess]], %[[SZero]] : (!emitc.ptrdiff_t, !emitc.size_t, i1, !emitc.ptrdiff_t) -> !emitc.ptrdiff_t {
  // CHECK-NEXT:   %[[SHRSI:[^ ]*]] = bitwise_right_shift %[[SC1]], %[[AmountIdx]] : (!emitc.ptrdiff_t, !emitc.size_t) -> !emitc.ptrdiff_t
  // CHECK-NEXT:   %[[STernary:[^ ]*]] = conditional %[[SCmpNoExcess]], %[[SHRSI]], %[[SZero]] : !emitc.ptrdiff_t
  // CHECK-NEXT:   yield %[[STernary]] : !emitc.ptrdiff_t
  // CHECK-NEXT: }
  // CHECK-NEXT: emitc.cast %[[SShiftRes]] : !emitc.ptrdiff_t to !emitc.size_t
  %3 = arith.shrsi %arg0, %arg1 : index

  return
}

// -----

func.func @arith_select(%arg0: i1, %arg1: tensor<8xi32>, %arg2: tensor<8xi32>) -> () {
  // CHECK: [[V0:[^ ]*]] = emitc.conditional %arg0, %arg1, %arg2 : tensor<8xi32>
  %0 = arith.select %arg0, %arg1, %arg2 : i1, tensor<8xi32>
  return
}

// -----

func.func @arith_cmpf_false(%arg0: f32, %arg1: f32) -> i1 {
  // CHECK-LABEL: arith_cmpf_false
  // CHECK-SAME: ([[Arg0:[^ ]*]]: f32, [[Arg1:[^ ]*]]: f32)
  // CHECK-DAG: [[False:[^ ]*]] = "emitc.constant"() <{value = false}> : () -> i1
  %false = arith.cmpf false, %arg0, %arg1 : f32
  // CHECK: return [[False]]
  return %false: i1
}

// -----

func.func @arith_cmpf_oeq(%arg0: f32, %arg1: f32) -> i1 {
  // CHECK-LABEL: arith_cmpf_oeq
  // CHECK-SAME: ([[Arg0:[^ ]*]]: f32, [[Arg1:[^ ]*]]: f32)
  // CHECK-DAG: [[EQ:[^ ]*]] = emitc.cmp eq, [[Arg0]], [[Arg1]] : (f32, f32) -> i1
  // CHECK-DAG: [[NotNaNArg0:[^ ]*]] = emitc.cmp eq, [[Arg0]], [[Arg0]] : (f32, f32) -> i1
  // CHECK-DAG: [[NotNaNArg1:[^ ]*]] = emitc.cmp eq, [[Arg1]], [[Arg1]] : (f32, f32) -> i1
  // CHECK-DAG: [[Ordered:[^ ]*]] = emitc.logical_and [[NotNaNArg0]], [[NotNaNArg1]] : i1, i1
  // CHECK-DAG: [[OEQ:[^ ]*]] = emitc.logical_and [[Ordered]], [[EQ]] : i1, i1
  %oeq = arith.cmpf oeq, %arg0, %arg1 : f32
  // CHECK: return [[OEQ]]
  return %oeq: i1
}

// -----

func.func @arith_cmpf_ogt(%arg0: f32, %arg1: f32) -> i1 {
  // CHECK-LABEL: arith_cmpf_ogt
  // CHECK-SAME: ([[Arg0:[^ ]*]]: f32, [[Arg1:[^ ]*]]: f32)
  // CHECK-DAG: [[GT:[^ ]*]] = emitc.cmp gt, [[Arg0]], [[Arg1]] : (f32, f32) -> i1
  // CHECK-DAG: [[NotNaNArg0:[^ ]*]] = emitc.cmp eq, [[Arg0]], [[Arg0]] : (f32, f32) -> i1
  // CHECK-DAG: [[NotNaNArg1:[^ ]*]] = emitc.cmp eq, [[Arg1]], [[Arg1]] : (f32, f32) -> i1
  // CHECK-DAG: [[Ordered:[^ ]*]] = emitc.logical_and [[NotNaNArg0]], [[NotNaNArg1]] : i1, i1
  // CHECK-DAG: [[OGT:[^ ]*]] = emitc.logical_and [[Ordered]], [[GT]] : i1, i1
  %ogt = arith.cmpf ogt, %arg0, %arg1 : f32
  // CHECK: return [[OGT]]
  return %ogt: i1
}

// -----

func.func @arith_cmpf_oge(%arg0: f32, %arg1: f32) -> i1 {
  // CHECK-LABEL: arith_cmpf_oge
  // CHECK-SAME: ([[Arg0:[^ ]*]]: f32, [[Arg1:[^ ]*]]: f32)
  // CHECK-DAG: [[GE:[^ ]*]] = emitc.cmp ge, [[Arg0]], [[Arg1]] : (f32, f32) -> i1
  // CHECK-DAG: [[NotNaNArg0:[^ ]*]] = emitc.cmp eq, [[Arg0]], [[Arg0]] : (f32, f32) -> i1
  // CHECK-DAG: [[NotNaNArg1:[^ ]*]] = emitc.cmp eq, [[Arg1]], [[Arg1]] : (f32, f32) -> i1
  // CHECK-DAG: [[Ordered:[^ ]*]] = emitc.logical_and [[NotNaNArg0]], [[NotNaNArg1]] : i1, i1
  // CHECK-DAG: [[OGE:[^ ]*]] = emitc.logical_and [[Ordered]], [[GE]] : i1, i1
  %oge = arith.cmpf oge, %arg0, %arg1 : f32
  // CHECK: return [[OGE]]
  return %oge: i1
}

// -----

func.func @arith_cmpf_olt(%arg0: f32, %arg1: f32) -> i1 {
  // CHECK-LABEL: arith_cmpf_olt
  // CHECK-SAME: ([[Arg0:[^ ]*]]: f32, [[Arg1:[^ ]*]]: f32)
  // CHECK-DAG: [[LT:[^ ]*]] = emitc.cmp lt, [[Arg0]], [[Arg1]] : (f32, f32) -> i1
  // CHECK-DAG: [[NotNaNArg0:[^ ]*]] = emitc.cmp eq, [[Arg0]], [[Arg0]] : (f32, f32) -> i1
  // CHECK-DAG: [[NotNaNArg1:[^ ]*]] = emitc.cmp eq, [[Arg1]], [[Arg1]] : (f32, f32) -> i1
  // CHECK-DAG: [[Ordered:[^ ]*]] = emitc.logical_and [[NotNaNArg0]], [[NotNaNArg1]] : i1, i1
  // CHECK-DAG: [[OLT:[^ ]*]] = emitc.logical_and [[Ordered]], [[LT]] : i1, i1
  %olt = arith.cmpf olt, %arg0, %arg1 : f32
  // CHECK: return [[OLT]]
  return %olt: i1
}

// -----

func.func @arith_cmpf_ole(%arg0: f32, %arg1: f32) -> i1 {
  // CHECK-LABEL: arith_cmpf_ole
  // CHECK-SAME: ([[Arg0:[^ ]*]]: f32, [[Arg1:[^ ]*]]: f32)
  // CHECK-DAG: [[LT:[^ ]*]] = emitc.cmp le, [[Arg0]], [[Arg1]] : (f32, f32) -> i1
  // CHECK-DAG: [[NotNaNArg0:[^ ]*]] = emitc.cmp eq, [[Arg0]], [[Arg0]] : (f32, f32) -> i1
  // CHECK-DAG: [[NotNaNArg1:[^ ]*]] = emitc.cmp eq, [[Arg1]], [[Arg1]] : (f32, f32) -> i1
  // CHECK-DAG: [[Ordered:[^ ]*]] = emitc.logical_and [[NotNaNArg0]], [[NotNaNArg1]] : i1, i1
  // CHECK-DAG: [[OLE:[^ ]*]] = emitc.logical_and [[Ordered]], [[LT]] : i1, i1
  %ole = arith.cmpf ole, %arg0, %arg1 : f32
  // CHECK: return [[OLE]]
  return %ole: i1
}

// -----

func.func @arith_cmpf_one(%arg0: f32, %arg1: f32) -> i1 {
  // CHECK-LABEL: arith_cmpf_one
  // CHECK-SAME: ([[Arg0:[^ ]*]]: f32, [[Arg1:[^ ]*]]: f32)
  // CHECK-DAG: [[NEQ:[^ ]*]] = emitc.cmp ne, [[Arg0]], [[Arg1]] : (f32, f32) -> i1
  // CHECK-DAG: [[NotNaNArg0:[^ ]*]] = emitc.cmp eq, [[Arg0]], [[Arg0]] : (f32, f32) -> i1
  // CHECK-DAG: [[NotNaNArg1:[^ ]*]] = emitc.cmp eq, [[Arg1]], [[Arg1]] : (f32, f32) -> i1
  // CHECK-DAG: [[Ordered:[^ ]*]] = emitc.logical_and [[NotNaNArg0]], [[NotNaNArg1]] : i1, i1
  // CHECK-DAG: [[ONE:[^ ]*]] = emitc.logical_and [[Ordered]], [[NEQ]] : i1, i1
  %one = arith.cmpf one, %arg0, %arg1 : f32
  // CHECK: return [[ONE]]
  return %one: i1
}

// -----

func.func @arith_cmpf_ord(%arg0: f32, %arg1: f32) -> i1 {
  // CHECK-LABEL: arith_cmpf_ord
  // CHECK-SAME: ([[Arg0:[^ ]*]]: f32, [[Arg1:[^ ]*]]: f32)
  // CHECK-DAG: [[NotNaNArg0:[^ ]*]] = emitc.cmp eq, [[Arg0]], [[Arg0]] : (f32, f32) -> i1
  // CHECK-DAG: [[NotNaNArg1:[^ ]*]] = emitc.cmp eq, [[Arg1]], [[Arg1]] : (f32, f32) -> i1
  // CHECK-DAG: [[Ordered:[^ ]*]] = emitc.logical_and [[NotNaNArg0]], [[NotNaNArg1]] : i1, i1
  %ord = arith.cmpf ord, %arg0, %arg1 : f32
  // CHECK: return [[Ordered]]
  return %ord: i1
}

// -----

func.func @arith_cmpf_ueq(%arg0: f32, %arg1: f32) -> i1 {
  // CHECK-LABEL: arith_cmpf_ueq
  // CHECK-SAME: ([[Arg0:[^ ]*]]: f32, [[Arg1:[^ ]*]]: f32)
  // CHECK-DAG: [[EQ:[^ ]*]] = emitc.cmp eq, [[Arg0]], [[Arg1]] : (f32, f32) -> i1
  // CHECK-DAG: [[NaNArg0:[^ ]*]] = emitc.cmp ne, [[Arg0]], [[Arg0]] : (f32, f32) -> i1
  // CHECK-DAG: [[NaNArg1:[^ ]*]] = emitc.cmp ne, [[Arg1]], [[Arg1]] : (f32, f32) -> i1
  // CHECK-DAG: [[Unordered:[^ ]*]] = emitc.logical_or [[NaNArg0]], [[NaNArg1]] : i1, i1
  // CHECK-DAG: [[UEQ:[^ ]*]] = emitc.logical_or [[Unordered]], [[EQ]] : i1, i1
  %ueq = arith.cmpf ueq, %arg0, %arg1 : f32
  // CHECK: return [[UEQ]]
  return %ueq: i1
}

// -----

func.func @arith_cmpf_ugt(%arg0: f32, %arg1: f32) -> i1 {
  // CHECK-LABEL: arith_cmpf_ugt
  // CHECK-SAME: ([[Arg0:[^ ]*]]: f32, [[Arg1:[^ ]*]]: f32)
  // CHECK-DAG: [[GT:[^ ]*]] = emitc.cmp gt, [[Arg0]], [[Arg1]] : (f32, f32) -> i1
  // CHECK-DAG: [[NaNArg0:[^ ]*]] = emitc.cmp ne, [[Arg0]], [[Arg0]] : (f32, f32) -> i1
  // CHECK-DAG: [[NaNArg1:[^ ]*]] = emitc.cmp ne, [[Arg1]], [[Arg1]] : (f32, f32) -> i1
  // CHECK-DAG: [[Unordered:[^ ]*]] = emitc.logical_or [[NaNArg0]], [[NaNArg1]] : i1, i1
  // CHECK-DAG: [[UGT:[^ ]*]] = emitc.logical_or [[Unordered]], [[GT]] : i1, i1
  %ugt = arith.cmpf ugt, %arg0, %arg1 : f32
  // CHECK: return [[UGT]]
  return %ugt: i1
}

// -----

func.func @arith_cmpf_uge(%arg0: f32, %arg1: f32) -> i1 {
  // CHECK-LABEL: arith_cmpf_uge
  // CHECK-SAME: ([[Arg0:[^ ]*]]: f32, [[Arg1:[^ ]*]]: f32)
  // CHECK-DAG: [[GE:[^ ]*]] = emitc.cmp ge, [[Arg0]], [[Arg1]] : (f32, f32) -> i1
  // CHECK-DAG: [[NaNArg0:[^ ]*]] = emitc.cmp ne, [[Arg0]], [[Arg0]] : (f32, f32) -> i1
  // CHECK-DAG: [[NaNArg1:[^ ]*]] = emitc.cmp ne, [[Arg1]], [[Arg1]] : (f32, f32) -> i1
  // CHECK-DAG: [[Unordered:[^ ]*]] = emitc.logical_or [[NaNArg0]], [[NaNArg1]] : i1, i1
  // CHECK-DAG: [[UGE:[^ ]*]] = emitc.logical_or [[Unordered]], [[GE]] : i1, i1
  %uge = arith.cmpf uge, %arg0, %arg1 : f32
  // CHECK: return [[UGE]]
  return %uge: i1
}

// -----

func.func @arith_cmpf_ult(%arg0: f32, %arg1: f32) -> i1 {
  // CHECK-LABEL: arith_cmpf_ult
  // CHECK-SAME: ([[Arg0:[^ ]*]]: f32, [[Arg1:[^ ]*]]: f32)
  // CHECK-DAG: [[LT:[^ ]*]] = emitc.cmp lt, [[Arg0]], [[Arg1]] : (f32, f32) -> i1
  // CHECK-DAG: [[NaNArg0:[^ ]*]] = emitc.cmp ne, [[Arg0]], [[Arg0]] : (f32, f32) -> i1
  // CHECK-DAG: [[NaNArg1:[^ ]*]] = emitc.cmp ne, [[Arg1]], [[Arg1]] : (f32, f32) -> i1
  // CHECK-DAG: [[Unordered:[^ ]*]] = emitc.logical_or [[NaNArg0]], [[NaNArg1]] : i1, i1
  // CHECK-DAG: [[ULT:[^ ]*]] = emitc.logical_or [[Unordered]], [[LT]] : i1, i1
  %ult = arith.cmpf ult, %arg0, %arg1 : f32
  // CHECK: return [[ULT]]
  return %ult: i1
}

// -----

func.func @arith_cmpf_ule(%arg0: f32, %arg1: f32) -> i1 {
  // CHECK-LABEL: arith_cmpf_ule
  // CHECK-SAME: ([[Arg0:[^ ]*]]: f32, [[Arg1:[^ ]*]]: f32)
  // CHECK-DAG: [[LE:[^ ]*]] = emitc.cmp le, [[Arg0]], [[Arg1]] : (f32, f32) -> i1
  // CHECK-DAG: [[NaNArg0:[^ ]*]] = emitc.cmp ne, [[Arg0]], [[Arg0]] : (f32, f32) -> i1
  // CHECK-DAG: [[NaNArg1:[^ ]*]] = emitc.cmp ne, [[Arg1]], [[Arg1]] : (f32, f32) -> i1
  // CHECK-DAG: [[Unordered:[^ ]*]] = emitc.logical_or [[NaNArg0]], [[NaNArg1]] : i1, i1
  // CHECK-DAG: [[ULE:[^ ]*]] = emitc.logical_or [[Unordered]], [[LE]] : i1, i1
  %ule = arith.cmpf ule, %arg0, %arg1 : f32
  // CHECK: return [[ULE]]
  return %ule: i1
}

// -----

func.func @arith_cmpf_une(%arg0: f32, %arg1: f32) -> i1 {
  // CHECK-LABEL: arith_cmpf_une
  // CHECK-SAME: ([[Arg0:[^ ]*]]: f32, [[Arg1:[^ ]*]]: f32)
  // CHECK-DAG: [[NEQ:[^ ]*]] = emitc.cmp ne, [[Arg0]], [[Arg1]] : (f32, f32) -> i1
  // CHECK-DAG: [[NaNArg0:[^ ]*]] = emitc.cmp ne, [[Arg0]], [[Arg0]] : (f32, f32) -> i1
  // CHECK-DAG: [[NaNArg1:[^ ]*]] = emitc.cmp ne, [[Arg1]], [[Arg1]] : (f32, f32) -> i1
  // CHECK-DAG: [[Unordered:[^ ]*]] = emitc.logical_or [[NaNArg0]], [[NaNArg1]] : i1, i1
  // CHECK-DAG: [[UNE:[^ ]*]] = emitc.logical_or [[Unordered]], [[NEQ]] : i1, i1
  %une = arith.cmpf une, %arg0, %arg1 : f32
  // CHECK: return [[UNE]]
  return %une: i1
}

// -----

func.func @arith_cmpf_uno(%arg0: f32, %arg1: f32) -> i1 {
  // CHECK-LABEL: arith_cmpf_uno
  // CHECK-SAME: ([[Arg0:[^ ]*]]: f32, [[Arg1:[^ ]*]]: f32)
  // CHECK-DAG: [[NaNArg0:[^ ]*]] = emitc.cmp ne, [[Arg0]], [[Arg0]] : (f32, f32) -> i1
  // CHECK-DAG: [[NaNArg1:[^ ]*]] = emitc.cmp ne, [[Arg1]], [[Arg1]] : (f32, f32) -> i1
  // CHECK-DAG: [[Unordered:[^ ]*]] = emitc.logical_or [[NaNArg0]], [[NaNArg1]] : i1, i1
  %uno = arith.cmpf uno, %arg0, %arg1 : f32
  // CHECK: return [[Unordered]]
  return %uno: i1
}

// -----

func.func @arith_cmpf_true(%arg0: f32, %arg1: f32) -> i1 {
  // CHECK-LABEL: arith_cmpf_true
  // CHECK-SAME: ([[Arg0:[^ ]*]]: f32, [[Arg1:[^ ]*]]: f32)
  // CHECK-DAG: [[True:[^ ]*]] = "emitc.constant"() <{value = true}> : () -> i1
  %ueq = arith.cmpf true, %arg0, %arg1 : f32
  // CHECK: return [[True]]
  return %ueq: i1
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

func.func @arith_cmpi_index(%arg0: i32, %arg1: i32) -> i1 {
  // CHECK-LABEL: arith_cmpi_index

  // CHECK: %[[Cst0:.*]] = emitc.cast %{{.*}} : {{.*}} to !emitc.size_t
  %idx0 = arith.index_cast %arg0 : i32 to index
  // CHECK: %[[Cst1:.*]] = emitc.cast %{{.*}} : {{.*}} to !emitc.size_t
  %idx1 = arith.index_cast %arg0 : i32 to index

  // CHECK-DAG: [[ULT:[^ ]*]] = emitc.cmp lt, %[[Cst0]], %[[Cst1]] : (!emitc.size_t, !emitc.size_t) -> i1
  %ult = arith.cmpi ult, %idx0, %idx1 : index

  // CHECK-DAG: %[[CastArg0:[^ ]*]] = emitc.cast %[[Cst0]] : !emitc.size_t to !emitc.ptrdiff_t
  // CHECK-DAG: %[[CastArg1:[^ ]*]] = emitc.cast %[[Cst1]] : !emitc.size_t to !emitc.ptrdiff_t
  // CHECK-DAG: %[[SLT:[^ ]*]] = emitc.cmp lt, %[[CastArg0]], %[[CastArg1]] : (!emitc.ptrdiff_t, !emitc.ptrdiff_t) -> i1
  %slt = arith.cmpi slt, %idx0, %idx1 : index

  // CHECK: return %[[SLT]]
  return %slt: i1
}


// -----

func.func @arith_negf(%arg0: f32) -> f32 {
  // CHECK-LABEL: arith_negf
  // CHECK-SAME: %[[Arg0:[^ ]*]]: f32
  // CHECK: %[[N:[^ ]*]] = emitc.unary_minus %[[Arg0]] : (f32) -> f32
  %n = arith.negf %arg0 : f32
  // CHECK: return %[[N]]
  return %n: f32
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

  // CHECK: %[[CAST0:.*]] = emitc.cast %arg0 : f32 to ui16
  // CHECK: emitc.cast %[[CAST0]] : ui16 to i16
  %5 = arith.fptoui %arg0 : f32 to i16

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

  // CHECK: %[[Const:.*]] = "emitc.constant"
  // CHECK-SAME: value = 1
  // CHECK-SAME: () -> i32
  // CHECK: %[[AndOne:.*]] = emitc.bitwise_and %[[Arg0]], %[[Const]] : (i32, i32) -> i32
  // CHECK: %[[Conv:.*]] = emitc.cast %[[AndOne]] : i32 to i1
  %bool = arith.trunci %arg0 : i32 to i1

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

// -----

func.func @arith_index_cast(%arg0: i32) -> i32 {
  // CHECK-LABEL: arith_index_cast
  // CHECK-SAME: (%[[Arg0:[^ ]*]]: i32)
  // CHECK: %[[Conv0:.*]] = emitc.cast %[[Arg0]] : i32 to !emitc.ptrdiff_t
  // CHECK: %[[Conv1:.*]] = emitc.cast %[[Conv0]] : !emitc.ptrdiff_t to !emitc.size_t
  %idx = arith.index_cast %arg0 : i32 to index
  // CHECK: %[[Conv2:.*]] = emitc.cast %[[Conv1]] : !emitc.size_t to !emitc.ptrdiff_t
  // CHECK: %[[Conv3:.*]] = emitc.cast %[[Conv2]] : !emitc.ptrdiff_t to i32
  %int = arith.index_cast %idx : index to i32

  // CHECK: %[[Const:.*]] = "emitc.constant"
  // CHECK-SAME: value = 1
  // CHECK-SAME: () -> !emitc.size_t
  // CHECK: %[[AndOne:.*]] = emitc.bitwise_and %[[Conv1]], %[[Const]] : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
  // CHECK: %[[Conv4:.*]] = emitc.cast %[[AndOne]] : !emitc.size_t to i1
  %bool = arith.index_cast %idx : index to i1

  return %int : i32
}

// -----

func.func @arith_index_castui(%arg0: i32) -> i32 {
  // CHECK-LABEL: arith_index_castui
  // CHECK-SAME: (%[[Arg0:[^ ]*]]: i32)
  // CHECK: %[[Conv0:.*]] = emitc.cast %[[Arg0]] : i32 to ui32
  // CHECK: %[[Conv1:.*]] = emitc.cast %[[Conv0]] : ui32 to !emitc.size_t
  %idx = arith.index_castui %arg0 : i32 to index
  // CHECK: %[[Conv2:.*]] = emitc.cast %[[Conv1]] : !emitc.size_t to ui32
  // CHECK: %[[Conv3:.*]] = emitc.cast %[[Conv2]] : ui32 to i32
  %int = arith.index_castui %idx : index to i32

  // CHECK: %[[Const:.*]] = "emitc.constant"
  // CHECK-SAME: value = 1
  // CHECK-SAME: () -> !emitc.size_t
  // CHECK: %[[AndOne:.*]] = emitc.bitwise_and %[[Conv1]], %[[Const]] : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
  // CHECK: %[[Conv4:.*]] = emitc.cast %[[AndOne]] : !emitc.size_t to i1
  %bool = arith.index_castui %idx : index to i1

  return %int : i32
}

// -----

func.func @arith_divui_remui(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK-LABEL: arith_divui_remui
  // CHECK-SAME: (%[[Arg0:[^ ]*]]: i32, %[[Arg1:[^ ]*]]: i32)
  // CHECK: %[[Conv0:.*]] = emitc.cast %[[Arg0]] : i32 to ui32
  // CHECK: %[[Conv1:.*]] = emitc.cast %[[Arg1]] : i32 to ui32
  // CHECK: %[[Div:.*]] = emitc.div %[[Conv0]], %[[Conv1]] : (ui32, ui32) -> ui32
  %div = arith.divui %arg0, %arg1 : i32

  // CHECK: %[[Conv2:.*]] = emitc.cast %[[Arg0]] : i32 to ui32
  // CHECK: %[[Conv3:.*]] = emitc.cast %[[Arg1]] : i32 to ui32
  // CHECK: %[[Rem:.*]] = emitc.rem %[[Conv2]], %[[Conv3]] : (ui32, ui32) -> ui32
  %rem = arith.remui %arg0, %arg1 : i32

  return %div : i32
}

// -----

func.func @arith_extf(%arg0: f16) -> f64 {
  // CHECK-LABEL: arith_extf
  // CHECK-SAME: (%[[Arg0:[^ ]*]]: f16)
  // CHECK: %[[Extd0:.*]] = emitc.cast %[[Arg0]] : f16 to f32
  %extd0 = arith.extf %arg0 : f16 to f32
  // CHECK: %[[Extd1:.*]] = emitc.cast %[[Extd0]] : f32 to f64
  %extd1 = arith.extf %extd0 : f32 to f64

  return %extd1 : f64
}

// -----

func.func @arith_truncf(%arg0: f64) -> f16 {
  // CHECK-LABEL: arith_truncf
  // CHECK-SAME: (%[[Arg0:[^ ]*]]: f64)
  // CHECK: %[[Truncd0:.*]] = emitc.cast %[[Arg0]] : f64 to f32
  %truncd0 = arith.truncf %arg0 : f64 to f32
  // CHECK: %[[Truncd1:.*]] = emitc.cast %[[Truncd0]] : f32 to f16
  %truncd1 = arith.truncf %truncd0 : f32 to f16

  return %truncd1 : f16
}
