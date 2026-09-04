// RUN: mlir-opt -allow-unregistered-dialect -convert-scf-to-emitc %s | FileCheck %s
// RUN: mlir-opt -allow-unregistered-dialect -convert-to-emitc="filter-dialects=scf" %s | FileCheck %s

emitc.func @payload_one_result(%arg: i32) -> i32 {
  %result = add %arg, %arg : (i32, i32) -> i32
  return %result : i32
}

func.func @one_result() -> i32 {
  %init = emitc.literal "1.0" : i32
  %var  = emitc.literal "1.0" : i32
  %exit = emitc.literal "10.0" : i32

  %res = scf.while (%arg1 = %init) : (i32) -> i32 {
    %sum = emitc.add %arg1, %var : (i32, i32) -> i32
    %condition = emitc.cmp lt, %sum, %exit : (i32, i32) -> i1
    %next = emitc.add %arg1, %arg1 : (i32, i32) -> i32
    scf.condition(%condition) %next : i32
  } do {
  ^bb0(%arg2: i32):
    %next_arg1 = emitc.call @payload_one_result(%arg2) : (i32) -> i32
    scf.yield %next_arg1 : i32
  }
  
  return %res : i32
}
// CHECK-LABEL:   emitc.func @payload_one_result(
// CHECK-SAME:      %[[ARG0:.*]]: i32) -> i32 {
// CHECK:           %[[VAL_0:.*]] = add %[[ARG0]], %[[ARG0]] : (i32, i32) -> i32
// CHECK:           return %[[VAL_0]] : i32
// CHECK:         }

// CHECK-LABEL:   func.func @one_result() -> i32 {
// CHECK:           %[[VAL_0:.*]] = emitc.literal "1.0" : i32
// CHECK:           %[[VAL_1:.*]] = emitc.literal "1.0" : i32
// CHECK:           %[[VAL_2:.*]] = emitc.literal "10.0" : i32
// CHECK:           %[[VAL_3:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
// CHECK:           %[[VAL_4:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
// CHECK:           emitc.assign %[[VAL_0]] : i32 to %[[VAL_4]] : <i32>
// CHECK:           %[[VAL_5:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i1>
// CHECK:           emitc.do {
// CHECK:             %[[VAL_6:.*]] = load %[[VAL_4]] : <i32>
// CHECK:             %[[VAL_7:.*]] = add %[[VAL_6]], %[[VAL_1]] : (i32, i32) -> i32
// CHECK:             %[[VAL_8:.*]] = cmp lt, %[[VAL_7]], %[[VAL_2]] : (i32, i32) -> i1
// CHECK:             %[[VAL_9:.*]] = add %[[VAL_6]], %[[VAL_6]] : (i32, i32) -> i32
// CHECK:             assign %[[VAL_9]] : i32 to %[[VAL_3]] : <i32>
// CHECK:             assign %[[VAL_8]] : i1 to %[[VAL_5]] : <i1>
// CHECK:             if %[[VAL_8]] {
// CHECK:               %[[VAL_10:.*]] = call @payload_one_result(%[[VAL_9]]) : (i32) -> i32
// CHECK:               assign %[[VAL_10]] : i32 to %[[VAL_4]] : <i32>
// CHECK:             }
// CHECK:           } while {
// CHECK:             %[[VAL_11:.*]] = expression %[[VAL_5]] : (!emitc.lvalue<i1>) -> i1 {
// CHECK:               %[[VAL_12:.*]] = load %[[VAL_5]] : <i1>
// CHECK:               yield %[[VAL_12]] : i1
// CHECK:             }
// CHECK:             yield %[[VAL_11]] : i1
// CHECK:           }
// CHECK:           %[[VAL_13:.*]] = emitc.load %[[VAL_3]] : <i32>
// CHECK:           return %[[VAL_13]] : i32
// CHECK:         }

emitc.func @payload_two_results(%arg: i32) -> i32 {
  %result = add %arg, %arg : (i32, i32) -> i32
  return %result : i32
}

func.func @two_results() -> i32 {
  %init = emitc.literal "1.0" : i32
  %exit = emitc.literal "10.0" : i32

  %res1, %res2 = scf.while (%arg1_1 = %init, %arg1_2 = %init) : (i32, i32) -> (i32, i32) {
    %sum = emitc.add %arg1_1, %arg1_2 : (i32, i32) -> i32
    %condition = emitc.cmp lt, %sum, %exit : (i32, i32) -> i1
    scf.condition(%condition) %init, %arg1_2  : i32, i32
  } do {
  ^bb0(%arg2_1 : i32, %arg2_2 : i32):
    %next1 = emitc.call @payload_two_results(%arg2_1) : (i32) -> i32
    %next2 = emitc.call @payload_two_results(%arg2_2) : (i32) -> i32
    scf.yield %next1, %next2 : i32, i32
  }
  
  return %res1 : i32
}
// CHECK-LABEL:   emitc.func @payload_two_results(
// CHECK-SAME:      %[[ARG0:.*]]: i32) -> i32 {
// CHECK:           %[[VAL_0:.*]] = add %[[ARG0]], %[[ARG0]] : (i32, i32) -> i32
// CHECK:           return %[[VAL_0]] : i32
// CHECK:         }

// CHECK-LABEL:   func.func @two_results() -> i32 {
// CHECK:           %[[VAL_0:.*]] = emitc.literal "1.0" : i32
// CHECK:           %[[VAL_1:.*]] = emitc.literal "10.0" : i32
// CHECK:           %[[VAL_2:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
// CHECK:           %[[VAL_3:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
// CHECK:           %[[VAL_4:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
// CHECK:           emitc.assign %[[VAL_0]] : i32 to %[[VAL_4]] : <i32>
// CHECK:           %[[VAL_5:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
// CHECK:           emitc.assign %[[VAL_0]] : i32 to %[[VAL_5]] : <i32>
// CHECK:           %[[VAL_6:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i1>
// CHECK:           emitc.do {
// CHECK:             %[[VAL_7:.*]] = load %[[VAL_4]] : <i32>
// CHECK:             %[[VAL_8:.*]] = load %[[VAL_5]] : <i32>
// CHECK:             %[[VAL_9:.*]] = add %[[VAL_7]], %[[VAL_8]] : (i32, i32) -> i32
// CHECK:             %[[VAL_10:.*]] = cmp lt, %[[VAL_9]], %[[VAL_1]] : (i32, i32) -> i1
// CHECK:             assign %[[VAL_0]] : i32 to %[[VAL_2]] : <i32>
// CHECK:             assign %[[VAL_8]] : i32 to %[[VAL_3]] : <i32>
// CHECK:             assign %[[VAL_10]] : i1 to %[[VAL_6]] : <i1>
// CHECK:             if %[[VAL_10]] {
// CHECK:               %[[VAL_11:.*]] = call @payload_two_results(%[[VAL_0]]) : (i32) -> i32
// CHECK:               %[[VAL_12:.*]] = call @payload_two_results(%[[VAL_8]]) : (i32) -> i32
// CHECK:               assign %[[VAL_11]] : i32 to %[[VAL_4]] : <i32>
// CHECK:               assign %[[VAL_12]] : i32 to %[[VAL_5]] : <i32>
// CHECK:             }
// CHECK:           } while {
// CHECK:             %[[VAL_13:.*]] = expression %[[VAL_6]] : (!emitc.lvalue<i1>) -> i1 {
// CHECK:               %[[VAL_14:.*]] = load %[[VAL_6]] : <i1>
// CHECK:               yield %[[VAL_14]] : i1
// CHECK:             }
// CHECK:             yield %[[VAL_13]] : i1
// CHECK:           }
// CHECK:           %[[VAL_15:.*]] = emitc.load %[[VAL_2]] : <i32>
// CHECK:           %[[VAL_16:.*]] = emitc.load %[[VAL_3]] : <i32>
// CHECK:           return %[[VAL_15]] : i32
// CHECK:         }

emitc.func @payload_double_use(%arg: i32) -> i32 {
  %result = add %arg, %arg : (i32, i32) -> i32
  return %result : i32
}

emitc.func @foo_with_side_effect(%arg: i32, %p : !emitc.ptr<i32>) -> i32 {
  %sum = add %arg, %arg : (i32, i32) -> i32
  emitc.verbatim "{}[0] = {};" args %p, %sum : !emitc.ptr<i32>, i32
  return %sum : i32
}

func.func @double_use(%p : !emitc.ptr<i32>) -> i32 {
  %init = emitc.literal "1.0" : i32
  %var  = emitc.literal "1.0" : i32
  %exit = emitc.literal "10.0" : i32
  %res = scf.while (%arg1 = %init) : (i32) -> i32 {
    %used_twice = emitc.call @foo_with_side_effect(%arg1, %p) : (i32, !emitc.ptr<i32>) -> i32
    %prod = emitc.add %used_twice, %used_twice : (i32, i32) -> i32
    %sum = emitc.add %arg1, %prod : (i32, i32) -> i32
    %condition = emitc.cmp lt, %sum, %exit : (i32, i32) -> i1
    scf.condition(%condition) %arg1 : i32
  } do {
  ^bb0(%arg2: i32):
    %next_arg1 = emitc.call @payload_double_use(%arg2) : (i32) -> i32
    scf.yield %next_arg1 : i32
  }
  return %res : i32
}
// CHECK-LABEL:   emitc.func @payload_double_use(
// CHECK-SAME:      %[[ARG0:.*]]: i32) -> i32 {
// CHECK:           %[[VAL_0:.*]] = add %[[ARG0]], %[[ARG0]] : (i32, i32) -> i32
// CHECK:           return %[[VAL_0]] : i32
// CHECK:         }

// CHECK-LABEL:   emitc.func @foo_with_side_effect(
// CHECK-SAME:      %[[ARG0:.*]]: i32,
// CHECK-SAME:      %[[ARG1:.*]]: !emitc.ptr<i32>) -> i32 {
// CHECK:           %[[VAL_0:.*]] = add %[[ARG0]], %[[ARG0]] : (i32, i32) -> i32
// CHECK:           verbatim "{}[0] = {};" args %[[ARG1]], %[[VAL_0]] : !emitc.ptr<i32>, i32
// CHECK:           return %[[VAL_0]] : i32
// CHECK:         }

// CHECK-LABEL:   func.func @double_use(
// CHECK-SAME:      %[[ARG0:.*]]: !emitc.ptr<i32>) -> i32 {
// CHECK:           %[[VAL_0:.*]] = emitc.literal "1.0" : i32
// CHECK:           %[[VAL_1:.*]] = emitc.literal "1.0" : i32
// CHECK:           %[[VAL_2:.*]] = emitc.literal "10.0" : i32
// CHECK:           %[[VAL_3:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
// CHECK:           %[[VAL_4:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
// CHECK:           emitc.assign %[[VAL_0]] : i32 to %[[VAL_4]] : <i32>
// CHECK:           %[[VAL_5:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i1>
// CHECK:           emitc.do {
// CHECK:             %[[VAL_6:.*]] = load %[[VAL_4]] : <i32>
// CHECK:             %[[VAL_7:.*]] = call @foo_with_side_effect(%[[VAL_6]], %[[ARG0]]) : (i32, !emitc.ptr<i32>) -> i32
// CHECK:             %[[VAL_8:.*]] = add %[[VAL_7]], %[[VAL_7]] : (i32, i32) -> i32
// CHECK:             %[[VAL_9:.*]] = add %[[VAL_6]], %[[VAL_8]] : (i32, i32) -> i32
// CHECK:             %[[VAL_10:.*]] = cmp lt, %[[VAL_9]], %[[VAL_2]] : (i32, i32) -> i1
// CHECK:             assign %[[VAL_6]] : i32 to %[[VAL_3]] : <i32>
// CHECK:             assign %[[VAL_10]] : i1 to %[[VAL_5]] : <i1>
// CHECK:             if %[[VAL_10]] {
// CHECK:               %[[VAL_11:.*]] = call @payload_double_use(%[[VAL_6]]) : (i32) -> i32
// CHECK:               assign %[[VAL_11]] : i32 to %[[VAL_4]] : <i32>
// CHECK:             }
// CHECK:           } while {
// CHECK:             %[[VAL_12:.*]] = expression %[[VAL_5]] : (!emitc.lvalue<i1>) -> i1 {
// CHECK:               %[[VAL_13:.*]] = load %[[VAL_5]] : <i1>
// CHECK:               yield %[[VAL_13]] : i1
// CHECK:             }
// CHECK:             yield %[[VAL_12]] : i1
// CHECK:           }
// CHECK:           %[[VAL_14:.*]] = emitc.load %[[VAL_3]] : <i32>
// CHECK:           return %[[VAL_14]] : i32
// CHECK:         }

emitc.func @payload_empty_after_region() -> i1 {
  %true = emitc.literal "true" : i1
  return %true : i1
}

func.func @empty_after_region() {
  scf.while () : () -> () {
    %condition = emitc.call @payload_empty_after_region() : () -> i1
    scf.condition(%condition)
  } do {
  ^bb0():
    scf.yield
  }
  return
}
// CHECK-LABEL:   emitc.func @payload_empty_after_region() -> i1 {
// CHECK:           %[[VAL_0:.*]] = literal "true" : i1
// CHECK:           return %[[VAL_0]] : i1
// CHECK:         }

// CHECK-LABEL:   func.func @empty_after_region() {
// CHECK:           %[[VAL_0:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i1>
// CHECK:           emitc.do {
// CHECK:             %[[VAL_1:.*]] = call @payload_empty_after_region() : () -> i1
// CHECK:             assign %[[VAL_1]] : i1 to %[[VAL_0]] : <i1>
// CHECK:           } while {
// CHECK:             %[[VAL_2:.*]] = expression %[[VAL_0]] : (!emitc.lvalue<i1>) -> i1 {
// CHECK:               %[[VAL_3:.*]] = load %[[VAL_0]] : <i1>
// CHECK:               yield %[[VAL_3]] : i1
// CHECK:             }
// CHECK:             yield %[[VAL_2]] : i1
// CHECK:           }
// CHECK:           return
// CHECK:         }

emitc.func @payload_different_number_of_vars(%arg0: i32) -> i32 {
  %0 = add %arg0, %arg0 : (i32, i32) -> i32
  return %0 : i32
}
func.func @different_number_of_vars() -> (i32, i32) {
  %init = emitc.literal "1.0" : i32
  %var  = emitc.literal "7.0" : i32
  %exit = emitc.literal "10.0" : i32
  %res, %res2 = scf.while (%arg1 = %init) : (i32) -> (i32, i32) {
    %sum = emitc.add %arg1, %var : (i32, i32) -> i32
    %condition = emitc.cmp lt, %sum, %exit : (i32, i32) -> i1
    %next = emitc.add %arg1, %arg1 : (i32, i32) -> i32
    scf.condition(%condition) %next, %sum : i32, i32
  } do {
  ^bb0(%arg2: i32, %arg3 : i32):
    %next_arg1 = emitc.call @payload_different_number_of_vars(%arg2) : (i32) -> i32
    scf.yield %next_arg1 : i32
  }
  return %res, %res2 : i32, i32
}
// CHECK-LABEL:   emitc.func @payload_different_number_of_vars(
// CHECK-SAME:      %[[ARG0:.*]]: i32) -> i32 {
// CHECK:           %[[VAL_0:.*]] = add %[[ARG0]], %[[ARG0]] : (i32, i32) -> i32
// CHECK:           return %[[VAL_0]] : i32
// CHECK:         }

// CHECK-LABEL:   func.func @different_number_of_vars() -> (i32, i32) {
// CHECK:           %[[VAL_0:.*]] = emitc.literal "1.0" : i32
// CHECK:           %[[VAL_1:.*]] = emitc.literal "7.0" : i32
// CHECK:           %[[VAL_2:.*]] = emitc.literal "10.0" : i32
// CHECK:           %[[VAL_3:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
// CHECK:           %[[VAL_4:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
// CHECK:           %[[VAL_5:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
// CHECK:           emitc.assign %[[VAL_0]] : i32 to %[[VAL_5]] : <i32>
// CHECK:           %[[VAL_6:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i1>
// CHECK:           emitc.do {
// CHECK:             %[[VAL_7:.*]] = load %[[VAL_5]] : <i32>
// CHECK:             %[[VAL_8:.*]] = add %[[VAL_7]], %[[VAL_1]] : (i32, i32) -> i32
// CHECK:             %[[VAL_9:.*]] = cmp lt, %[[VAL_8]], %[[VAL_2]] : (i32, i32) -> i1
// CHECK:             %[[VAL_10:.*]] = add %[[VAL_7]], %[[VAL_7]] : (i32, i32) -> i32
// CHECK:             assign %[[VAL_10]] : i32 to %[[VAL_3]] : <i32>
// CHECK:             assign %[[VAL_8]] : i32 to %[[VAL_4]] : <i32>
// CHECK:             assign %[[VAL_9]] : i1 to %[[VAL_6]] : <i1>
// CHECK:             if %[[VAL_9]] {
// CHECK:               %[[VAL_11:.*]] = call @payload_different_number_of_vars(%[[VAL_10]]) : (i32) -> i32
// CHECK:               assign %[[VAL_11]] : i32 to %[[VAL_5]] : <i32>
// CHECK:             }
// CHECK:           } while {
// CHECK:             %[[VAL_12:.*]] = expression %[[VAL_6]] : (!emitc.lvalue<i1>) -> i1 {
// CHECK:               %[[VAL_13:.*]] = load %[[VAL_6]] : <i1>
// CHECK:               yield %[[VAL_13]] : i1
// CHECK:             }
// CHECK:             yield %[[VAL_12]] : i1
// CHECK:           }
// CHECK:           %[[VAL_14:.*]] = emitc.load %[[VAL_3]] : <i32>
// CHECK:           %[[VAL_15:.*]] = emitc.load %[[VAL_4]] : <i32>
// CHECK:           return %[[VAL_14]], %[[VAL_15]] : i32, i32
// CHECK:         }
