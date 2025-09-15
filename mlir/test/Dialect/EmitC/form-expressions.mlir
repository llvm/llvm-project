// RUN: mlir-opt %s -form-expressions | FileCheck %s

// CHECK-LABEL: func.func @single_expression(
// CHECK-SAME:                               %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32) -> i1 {
// CHECK:           %[[VAL_5:.*]] = emitc.expression %[[VAL_3]], %[[VAL_2]], %[[VAL_0]] : (i32, i32, i32) -> i1 {
// CHECK:             %[[VAL_4:.*]] = "emitc.constant"() <{value = 42 : i32}> : () -> i32
// CHECK:             %[[VAL_6:.*]] = mul %[[VAL_0]], %[[VAL_4]] : (i32, i32) -> i32
// CHECK:             %[[VAL_7:.*]] = sub %[[VAL_6]], %[[VAL_2]] : (i32, i32) -> i32
// CHECK:             %[[VAL_8:.*]] = cmp lt, %[[VAL_7]], %[[VAL_3]] : (i32, i32) -> i1
// CHECK:             yield %[[VAL_8]] : i1
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
// CHECK:         %[[VAL_4:.*]] = emitc.expression %[[VAL_2]], %[[VAL_0]], %[[VAL_1]] : (i32, i32, i32) -> i32 {
// CHECK:           %[[VAL_5:.*]] = mul %[[VAL_0]], %[[VAL_1]] : (i32, i32) -> i32
// CHECK:           %[[VAL_6:.*]] = sub %[[VAL_5]], %[[VAL_2]] : (i32, i32) -> i32
// CHECK:           yield %[[VAL_6]] : i32
// CHECK:         }
// CHECK:         %[[VAL_7:.*]] = emitc.expression %[[VAL_2]], %[[VAL_1]], %[[VAL_3]] : (i32, i32, i32) -> i32 {
// CHECK:           %[[VAL_8:.*]] = add %[[VAL_1]], %[[VAL_3]] : (i32, i32) -> i32
// CHECK:           %[[VAL_9:.*]] = div %[[VAL_8]], %[[VAL_2]] : (i32, i32) -> i32
// CHECK:           yield %[[VAL_9]] : i32
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
// CHECK:         %[[VAL_4:.*]] = emitc.expression %[[VAL_2]], %[[VAL_0]], %[[VAL_1]] : (i32, i32, i32) -> i32 {
// CHECK:           %[[VAL_5:.*]] = mul %[[VAL_0]], %[[VAL_1]] : (i32, i32) -> i32
// CHECK:           %[[VAL_6:.*]] = call_opaque "foo"(%[[VAL_5]], %[[VAL_2]]) : (i32, i32) -> i32
// CHECK:           yield %[[VAL_6]] : i32
// CHECK:         }
// CHECK:         %[[VAL_7:.*]] = emitc.expression %[[VAL_4]], %[[VAL_1]] : (i32, i32) -> i1 {
// CHECK:           %[[VAL_8:.*]] = cmp lt, %[[VAL_4]], %[[VAL_1]] : (i32, i32) -> i1
// CHECK:           yield %[[VAL_8]] : i1
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
// CHECK:         %[[VAL_3:.*]] = emitc.expression %[[VAL_2]] : (!emitc.ptr<i32>) -> i32 {
// CHECK:           %[[VAL_4:.*]] = apply "*"(%[[VAL_2]]) : (!emitc.ptr<i32>) -> i32
// CHECK:           yield %[[VAL_4]] : i32
// CHECK:         }
// CHECK:         %[[VAL_5:.*]] = emitc.expression %[[VAL_3]], %[[VAL_0]], %[[VAL_1]] : (i32, i32, i32) -> i1 {
// CHECK:           %[[VAL_6:.*]] = mul %[[VAL_0]], %[[VAL_1]] : (i32, i32) -> i32
// CHECK:           %[[VAL_7:.*]] = cmp lt, %[[VAL_6]], %[[VAL_3]] : (i32, i32) -> i1
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
// CHECK:         %[[VAL_3:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
// CHECK:         %[[VAL_4:.*]] = emitc.expression %[[VAL_2]], %[[VAL_1]], %[[VAL_3]] : (!emitc.ptr<i32>, i32, !emitc.lvalue<i32>) -> i1 {
// CHECK:           %[[VAL_5:.*]] = apply "&"(%[[VAL_3]]) : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>
// CHECK:           %[[VAL_6:.*]] = add %[[VAL_5]], %[[VAL_1]] : (!emitc.ptr<i32>, i32) -> !emitc.ptr<i32>
// CHECK:           %[[VAL_7:.*]] = cmp lt, %[[VAL_6]], %[[VAL_2]] : (!emitc.ptr<i32>, !emitc.ptr<i32>) -> i1
// CHECK:           yield %[[VAL_7]] : i1
// CHECK:         }
// CHECK:         return %[[VAL_4]] : i1
// CHECK:       }

func.func @expression_with_address_taken(%arg0: i32, %arg1: i32, %arg2: !emitc.ptr<i32>) -> i1 {
  %0 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
  %a = emitc.apply "&"(%0) : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>
  %b = emitc.add %a, %arg1 : (!emitc.ptr<i32>, i32) -> !emitc.ptr<i32>
  %c = emitc.cmp lt, %b, %arg2 :(!emitc.ptr<i32>, !emitc.ptr<i32>) -> i1
  return %c : i1
}

// CHECK-LABEL: func.func @no_nested_expression(
// CHECK-SAME:      %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32) -> i1 {
// CHECK:         %[[VAL_2:.*]] = emitc.expression %[[VAL_0]], %[[VAL_1]] : (i32, i32) -> i1 {
// CHECK:           %[[VAL_3:.*]] = cmp lt, %[[VAL_0]], %[[VAL_1]] : (i32, i32) -> i1
// CHECK:           yield %[[VAL_3]] : i1
// CHECK:         }
// CHECK:         return %[[VAL_2]] : i1
// CHECK:       }

func.func @no_nested_expression(%arg0: i32, %arg1: i32) -> i1 {
  %a = emitc.expression %arg0, %arg1 :(i32, i32) -> i1 {
    %b = emitc.cmp lt, %arg0, %arg1 :(i32, i32) -> i1
    emitc.yield %b : i1
  }
  return %a : i1
}

// CHECK-LABEL: func.func @single_result_requirement
// CHECK-NOT:  emitc.expression

func.func @single_result_requirement() -> (i32, i32) {
  %0:2 = emitc.call_opaque "foo" () : () -> (i32, i32)
  return %0#0, %0#1 : i32, i32
}

// CHECK-LABEL:   func.func @expression_with_load(
// CHECK-SAME:                                    %[[VAL_0:.*]]: i32,
// CHECK-SAME:                                    %[[VAL_1:.*]]: !emitc.ptr<i32>) -> i1 {
// CHECK:           %[[VAL_2:.*]] = emitc.expression : () -> i64 {
// CHECK:             %[[VAL_C:.*]] = "emitc.constant"() <{value = 0 : i64}> : () -> i64
// CHECK:             yield %[[VAL_C]] : i64
// CHECK:           }
// CHECK:           %[[VAL_3:.*]] = "emitc.variable"() <{value = #emitc.opaque<"42">}> : () -> !emitc.lvalue<i32>
// CHECK:           %[[VAL_4:.*]] = emitc.expression %[[VAL_3]] : (!emitc.lvalue<i32>) -> i32 {
// CHECK:             %[[VAL_5:.*]] = load %[[VAL_3]] : <i32>
// CHECK:             yield %[[VAL_5]] : i32
// CHECK:           }
// CHECK:           %[[VAL_6:.*]] = emitc.subscript %[[VAL_1]]{{\[}}%[[VAL_2]]] : (!emitc.ptr<i32>, i64) -> !emitc.lvalue<i32>
// CHECK:           %[[VAL_7:.*]] = emitc.expression %[[VAL_6]] : (!emitc.lvalue<i32>) -> i32 {
// CHECK:             %[[VAL_8:.*]] = load %[[VAL_6]] : <i32>
// CHECK:             yield %[[VAL_8]] : i32
// CHECK:           }
// CHECK:           %[[VAL_9:.*]] = emitc.expression %[[VAL_0]], %[[VAL_4]], %[[VAL_7]] : (i32, i32, i32) -> i1 {
// CHECK:             %[[VAL_10:.*]] = add %[[VAL_4]], %[[VAL_7]] : (i32, i32) -> i32
// CHECK:             %[[VAL_11:.*]] = cmp lt, %[[VAL_10]], %[[VAL_0]] : (i32, i32) -> i1
// CHECK:             yield %[[VAL_11]] : i1
// CHECK:           }
// CHECK:           return %[[VAL_9]] : i1
// CHECK:         }

func.func @expression_with_load(%arg0: i32, %arg1: !emitc.ptr<i32>) -> i1 {
  %c0 = "emitc.constant"() {value = 0 : i64} : () -> i64
  %0 = "emitc.variable"() <{value = #emitc.opaque<"42">}> : () -> !emitc.lvalue<i32>
  %a = emitc.load %0 : !emitc.lvalue<i32>
  %ptr = emitc.subscript %arg1[%c0] : (!emitc.ptr<i32>, i64) -> !emitc.lvalue<i32>
  %ptr_load = emitc.load %ptr : !emitc.lvalue<i32>
  %b = emitc.add %a, %ptr_load : (i32, i32) -> i32
  %c = emitc.cmp lt, %b, %arg0 :(i32, i32) -> i1
  return %c : i1
}

// CHECK-LABEL:   func.func @opaque_type_expression(%arg0: i32, %arg1: !emitc.opaque<"T0">, %arg2: i32) -> i1 {
// CHECK:           %0 = emitc.expression : () -> !emitc.opaque<"T1"> {
// CHECK:             %4 = "emitc.constant"() <{value = #emitc.opaque<"V">}> : () -> !emitc.opaque<"T1">
// CHECK:             yield %4 : !emitc.opaque<"T1">
// CHECK:           }
// CHECK:           %1 = emitc.expression %arg0, %0 : (i32, !emitc.opaque<"T1">) -> i32 {
// CHECK:             %4 = mul %arg0, %0 : (i32, !emitc.opaque<"T1">) -> i32
// CHECK:             yield %4 : i32
// CHECK:           }
// CHECK:           %2 = emitc.expression %1, %arg1 : (i32, !emitc.opaque<"T0">) -> i32 {
// CHECK:             %4 = sub %1, %arg1 : (i32, !emitc.opaque<"T0">) -> i32
// CHECK:             yield %4 : i32
// CHECK:           }
// CHECK:           %3 = emitc.expression %2, %arg2 : (i32, i32) -> i1 {
// CHECK:             %4 = cmp lt, %2, %arg2 : (i32, i32) -> i1
// CHECK:             yield %4 : i1
// CHECK:           }
// CHECK:           return %3 : i1
// CHECK:         }


func.func @opaque_type_expression(%arg0: i32,  %arg1: !emitc.opaque<"T0">, %arg2: i32) -> i1 {
  %c42 = "emitc.constant"(){value = #emitc.opaque<"V">} : () -> !emitc.opaque<"T1">
  %a = emitc.mul %arg0, %c42 : (i32, !emitc.opaque<"T1">) -> i32
  %b = emitc.sub %a, %arg1 : (i32, !emitc.opaque<"T0">) -> i32
  %c = emitc.cmp lt, %b, %arg2 :(i32, i32) -> i1
  return %c : i1
}

// CHECK-LABEL:   func.func @expression_with_constant(
// CHECK-SAME:                                        %[[VAL_0:.*]]: i32) -> i32 {
// CHECK:           %[[VAL_1:.*]] = emitc.expression %[[VAL_0]] : (i32) -> i32 {
// CHECK:             %[[VAL_2:.*]] = "emitc.constant"() <{value = 42 : i32}> : () -> i32
// CHECK:             %[[VAL_3:.*]] = mul %[[VAL_0]], %[[VAL_2]] : (i32, i32) -> i32
// CHECK:             yield %[[VAL_3]] : i32
// CHECK:           }
// CHECK:           return %[[VAL_1]] : i32
// CHECK:         }

func.func @expression_with_constant(%arg0: i32) -> i32 {
  %c42 = "emitc.constant"(){value = 42 : i32} : () -> i32
  %a = emitc.mul %arg0, %c42 : (i32, i32) -> i32
  return %a : i32
}
