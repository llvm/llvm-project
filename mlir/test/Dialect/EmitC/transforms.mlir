// RUN: mlir-opt %s --form-expressions --verify-diagnostics --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @single_expression(
// CHECK-SAME:                               %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32) -> i1 {
// CHECK:           %[[VAL_4:.*]] = "emitc.constant"() <{value = 42 : i32}> : () -> i32
// CHECK:           %[[VAL_5:.*]] = emitc.expression : i1 {
// CHECK:             %[[VAL_6:.*]] = emitc.mul %[[VAL_0]], %[[VAL_4]] : (i32, i32) -> i32
// CHECK:             %[[VAL_7:.*]] = emitc.sub %[[VAL_6]], %[[VAL_2]] : (i32, i32) -> i32
// CHECK:             %[[VAL_8:.*]] = emitc.cmp lt, %[[VAL_7]], %[[VAL_3]] : (i32, i32) -> i1
// CHECK:             emitc.yield %[[VAL_8]] : i1
// CHECK:           }
// CHECK:           return %[[VAL_5]] : i1
// CHECK:       }

func.func @single_expression(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32) -> i1 {
  %c42 = "emitc.constant"(){value = 42 : i32} : () -> i32
  %a = emitc.mul %arg0, %c42 : (i32, i32) -> i32
  %b = emitc.sub %a, %arg2 : (i32, i32) -> i32
  %c = emitc.cmp lt, %b, %arg3 :(i32, i32) -> i1
  return %c : i1
}

// CHECK-LABEL: func.func @multiple_expressions(
// CHECK-SAME:      %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32) -> (i32, i32) {
// CHECK:         %[[VAL_4:.*]] = emitc.expression : i32 {
// CHECK:           %[[VAL_5:.*]] = emitc.mul %[[VAL_0]], %[[VAL_1]] : (i32, i32) -> i32
// CHECK:           %[[VAL_6:.*]] = emitc.sub %[[VAL_5]], %[[VAL_2]] : (i32, i32) -> i32
// CHECK:           emitc.yield %[[VAL_6]] : i32
// CHECK:         }
// CHECK:         %[[VAL_7:.*]] = emitc.expression : i32 {
// CHECK:           %[[VAL_8:.*]] = emitc.add %[[VAL_1]], %[[VAL_3]] : (i32, i32) -> i32
// CHECK:           %[[VAL_9:.*]] = emitc.div %[[VAL_8]], %[[VAL_2]] : (i32, i32) -> i32
// CHECK:           emitc.yield %[[VAL_9]] : i32
// CHECK:         }
// CHECK:         return %[[VAL_4]], %[[VAL_7]] : i32, i32
// CHECK:       }

func.func @multiple_expressions(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32) -> (i32, i32) {
  %a = emitc.mul %arg0, %arg1 : (i32, i32) -> i32
  %b = emitc.sub %a, %arg2 : (i32, i32) -> i32
  %c = emitc.add %arg1, %arg3 : (i32, i32) -> i32
  %d = emitc.div %c, %arg2 : (i32, i32) -> i32
  return %b, %d : i32, i32
}

// CHECK-LABEL: func.func @expression_with_call(
// CHECK-SAME:      %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32) -> i1 {
// CHECK:         %[[VAL_4:.*]] = emitc.expression : i32 {
// CHECK:           %[[VAL_5:.*]] = emitc.mul %[[VAL_0]], %[[VAL_1]] : (i32, i32) -> i32
// CHECK:           %[[VAL_6:.*]] = emitc.call_opaque "foo"(%[[VAL_5]], %[[VAL_2]]) : (i32, i32) -> i32
// CHECK:           emitc.yield %[[VAL_6]] : i32
// CHECK:         }
// CHECK:         %[[VAL_7:.*]] = emitc.expression : i1 {
// CHECK:           %[[VAL_8:.*]] = emitc.cmp lt, %[[VAL_4]], %[[VAL_1]] : (i32, i32) -> i1
// CHECK:           emitc.yield %[[VAL_8]] : i1
// CHECK:         }
// CHECK:         return %[[VAL_7]] : i1
// CHECK:       }

func.func @expression_with_call(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32) -> i1 {
  %a = emitc.mul %arg0, %arg1 : (i32, i32) -> i32
  %b = emitc.call_opaque "foo" (%a, %arg2) : (i32, i32) -> (i32)
  %c = emitc.cmp lt, %b, %arg1 :(i32, i32) -> i1
  return %c : i1
}

// CHECK-LABEL: func.func @expression_with_dereference(
// CHECK-SAME:      %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: !emitc.ptr<i32>) -> i1 {
// CHECK:         %[[VAL_3:.*]] = emitc.expression : i32 {
// CHECK:           %[[VAL_4:.*]] = emitc.apply "*"(%[[VAL_2]]) : (!emitc.ptr<i32>) -> i32
// CHECK:           emitc.yield %[[VAL_4]] : i32
// CHECK:         }
// CHECK:         %[[VAL_5:.*]] = emitc.expression : i1 {
// CHECK:           %[[VAL_6:.*]] = emitc.mul %[[VAL_0]], %[[VAL_1]] : (i32, i32) -> i32
// CHECK:           %[[VAL_7:.*]] = emitc.cmp lt, %[[VAL_6]], %[[VAL_3]] : (i32, i32) -> i1
// CHECK:           emitc.yield %[[VAL_7]] : i1
// CHECK:         }
// CHECK:         return %[[VAL_5]] : i1
// CHECK:       }

func.func @expression_with_dereference(%arg0: i32, %arg1: i32, %arg2: !emitc.ptr<i32>) -> i1 {
  %a = emitc.mul %arg0, %arg1 : (i32, i32) -> i32
  %b = emitc.apply "*"(%arg2) : (!emitc.ptr<i32>) -> (i32)
  %c = emitc.cmp lt, %a, %b :(i32, i32) -> i1
  return %c : i1
}

// CHECK-LABEL: func.func @expression_with_address_taken(
// CHECK-SAME:      %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: !emitc.ptr<i32>) -> i1 {
// CHECK:         %[[VAL_3:.*]] = emitc.expression : i32 {
// CHECK:           %[[VAL_4:.*]] = emitc.rem %[[VAL_0]], %[[VAL_1]] : (i32, i32) -> i32
// CHECK:           emitc.yield %[[VAL_4]] : i32
// CHECK:         }
// CHECK:         %[[VAL_5:.*]] = emitc.expression : i1 {
// CHECK:           %[[VAL_6:.*]] = emitc.apply "&"(%[[VAL_3]]) : (i32) -> !emitc.ptr<i32>
// CHECK:           %[[VAL_7:.*]] = emitc.add %[[VAL_6]], %[[VAL_1]] : (!emitc.ptr<i32>, i32) -> !emitc.ptr<i32>
// CHECK:           %[[VAL_8:.*]] = emitc.cmp lt, %[[VAL_7]], %[[VAL_2]] : (!emitc.ptr<i32>, !emitc.ptr<i32>) -> i1
// CHECK:           emitc.yield %[[VAL_8]] : i1
// CHECK:         }
// CHECK:         return %[[VAL_5]] : i1
// CHECK:       }

func.func @expression_with_address_taken(%arg0: i32, %arg1: i32, %arg2: !emitc.ptr<i32>) -> i1 {
  %a = emitc.rem %arg0, %arg1 : (i32, i32) -> (i32)
  %b = emitc.apply "&"(%a) : (i32) -> !emitc.ptr<i32>
  %c = emitc.add %b, %arg1 : (!emitc.ptr<i32>, i32) -> !emitc.ptr<i32>
  %d = emitc.cmp lt, %c, %arg2 :(!emitc.ptr<i32>, !emitc.ptr<i32>) -> i1
  return %d : i1
}
