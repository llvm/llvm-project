// RUN: mlir-opt -allow-unregistered-dialect -convert-scf-to-emitc %s | FileCheck %s
// RUN: mlir-opt -allow-unregistered-dialect -convert-to-emitc="filter-dialects=scf" %s | FileCheck %s

emitc.func @payload_whileOp(%arg: i32) -> i32 {
  %result = emitc.add %arg, %arg : (i32, i32) -> i32
  return %result : i32
}

func.func @whileOp() -> i32 {
  %init = emitc.literal "1.0" : i32
  %var  = emitc.literal "1.0" : i32
  %exit = emitc.literal "10.0" : i32

  %res = scf.while (%arg1 = %init) : (i32) -> i32 {
    %sum = emitc.add %arg1, %var : (i32, i32) -> i32
    %condition = emitc.cmp lt, %sum, %exit : (i32, i32) -> i1
    scf.condition(%condition) %arg1 : i32
  } do {
  ^bb0(%arg2: i32):
    %next_arg1 = emitc.call @payload_whileOp(%arg2) : (i32) -> i32
    scf.yield %next_arg1 : i32
  }
  
  return %res : i32
}
// CHECK-LABEL:   emitc.func @payload_whileOp(
// CHECK-SAME:      %[[ARG0:.*]]: i32) -> i32 {
// CHECK:           %[[VAL_0:.*]] = add %[[ARG0]], %[[ARG0]] : (i32, i32) -> i32
// CHECK:           return %[[VAL_0]] : i32
// CHECK:         }

// CHECK-LABEL:   func.func @whileOp() -> i32 {
// CHECK:           %[[VAL_0:.*]] = emitc.literal "1.0" : i32
// CHECK:           %[[VAL_1:.*]] = emitc.literal "1.0" : i32
// CHECK:           %[[VAL_2:.*]] = emitc.literal "10.0" : i32
// CHECK:           %[[VAL_3:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
// CHECK:           emitc.assign %[[VAL_0]] : i32 to %[[VAL_3]] : <i32>
// CHECK:           emitc.while {
// CHECK:             %[[VAL_4:.*]] = load %[[VAL_3]] : <i32>
// CHECK:             %[[VAL_5:.*]] = add %[[VAL_4]], %[[VAL_1]] : (i32, i32) -> i32
// CHECK:             %[[VAL_6:.*]] = cmp lt, %[[VAL_5]], %[[VAL_2]] : (i32, i32) -> i1
// CHECK:             yield %[[VAL_6]] : i1
// CHECK:           } do {
// CHECK:             %[[VAL_7:.*]] = load %[[VAL_3]] : <i32>
// CHECK:             %[[VAL_8:.*]] = call @payload_whileOp(%[[VAL_7]]) : (i32) -> i32
// CHECK:             assign %[[VAL_8]] : i32 to %[[VAL_3]] : <i32>
// CHECK:           }
// CHECK:           %[[VAL_9:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
// CHECK:           %[[VAL_10:.*]] = emitc.load %[[VAL_3]] : <i32>
// CHECK:           emitc.assign %[[VAL_10]] : i32 to %[[VAL_9]] : <i32>
// CHECK:           %[[VAL_11:.*]] = emitc.load %[[VAL_9]] : <i32>
// CHECK:           return %[[VAL_11]] : i32
// CHECK:         }

emitc.func @payload_doOp(%arg: i32) -> i32 {
  %result = emitc.add %arg, %arg : (i32, i32) -> i32
  return %result : i32
}

func.func @doOp() -> i32 {
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
    %next_arg1 = emitc.call @payload_doOp(%arg2) : (i32) -> i32
    scf.yield %next_arg1 : i32
  }
  
  return %res : i32
}
// CHECK-LABEL:   emitc.func @payload_doOp(
// CHECK-SAME:      %[[ARG0:.*]]: i32) -> i32 {
// CHECK:           %[[VAL_0:.*]] = add %[[ARG0]], %[[ARG0]] : (i32, i32) -> i32
// CHECK:           return %[[VAL_0]] : i32
// CHECK:         }

// CHECK-LABEL:   func.func @doOp() -> i32 {
// CHECK:           %[[VAL_0:.*]] = emitc.literal "1.0" : i32
// CHECK:           %[[VAL_1:.*]] = emitc.literal "1.0" : i32
// CHECK:           %[[VAL_2:.*]] = emitc.literal "10.0" : i32
// CHECK:           %[[VAL_3:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
// CHECK:           emitc.assign %[[VAL_0]] : i32 to %[[VAL_3]] : <i32>
// CHECK:           %[[VAL_4:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i1>
// CHECK:           emitc.do {
// CHECK:             %[[VAL_5:.*]] = load %[[VAL_3]] : <i32>
// CHECK:             %[[VAL_6:.*]] = add %[[VAL_5]], %[[VAL_1]] : (i32, i32) -> i32
// CHECK:             %[[VAL_7:.*]] = cmp lt, %[[VAL_6]], %[[VAL_2]] : (i32, i32) -> i1
// CHECK:             %[[VAL_8:.*]] = add %[[VAL_5]], %[[VAL_5]] : (i32, i32) -> i32
// CHECK:             assign %[[VAL_7]] : i1 to %[[VAL_4]] : <i1>
// CHECK:             if %[[VAL_7]] {
// CHECK:               %[[VAL_9:.*]] = call @payload_doOp(%[[VAL_8]]) : (i32) -> i32
// CHECK:               assign %[[VAL_9]] : i32 to %[[VAL_3]] : <i32>
// CHECK:             }
// CHECK:           } while {
// CHECK:             %[[VAL_10:.*]] = load %[[VAL_4]] : <i1>
// CHECK:             yield %[[VAL_10]] : i1
// CHECK:           }
// CHECK:           %[[VAL_11:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
// CHECK:           %[[VAL_12:.*]] = emitc.load %[[VAL_3]] : <i32>
// CHECK:           emitc.assign %[[VAL_12]] : i32 to %[[VAL_11]] : <i32>
// CHECK:           %[[VAL_13:.*]] = emitc.load %[[VAL_11]] : <i32>
// CHECK:           return %[[VAL_13]] : i32
// CHECK:         }

emitc.func @payload_whileOp_two_results(%arg: i32) -> i32 {
  %result = emitc.add %arg, %arg : (i32, i32) -> i32
  return %result : i32
}

func.func @whileOp_two_results() -> i32 {
  %init = emitc.literal "1.0" : i32
  %exit = emitc.literal "10.0" : i32

  %res1, %res2 = scf.while (%arg1_1 = %init, %arg1_2 = %init) : (i32, i32) -> (i32, i32) {
    %sum = emitc.add %arg1_1, %arg1_2 : (i32, i32) -> i32
    %condition = emitc.cmp lt, %sum, %exit : (i32, i32) -> i1
    scf.condition(%condition) %arg1_1, %arg1_2  : i32, i32
  } do {
  ^bb0(%arg2_1 : i32, %arg2_2 : i32):
    %next1 = emitc.call @payload_whileOp_two_results(%arg2_1) : (i32) -> i32
    %next2 = emitc.call @payload_whileOp_two_results(%arg2_2) : (i32) -> i32
    scf.yield %next1, %next2 : i32, i32
  }
  
  return %res1 : i32
}

// CHECK-LABEL:   emitc.func @payload_whileOp_two_results(
// CHECK-SAME:      %[[ARG0:.*]]: i32) -> i32 {
// CHECK:           %[[VAL_0:.*]] = add %[[ARG0]], %[[ARG0]] : (i32, i32) -> i32
// CHECK:           return %[[VAL_0]] : i32
// CHECK:         }

// CHECK-LABEL:   func.func @whileOp_two_results() -> i32 {
// CHECK:           %[[VAL_0:.*]] = emitc.literal "1.0" : i32
// CHECK:           %[[VAL_1:.*]] = emitc.literal "10.0" : i32
// CHECK:           %[[VAL_2:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
// CHECK:           emitc.assign %[[VAL_0]] : i32 to %[[VAL_2]] : <i32>
// CHECK:           %[[VAL_3:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
// CHECK:           emitc.assign %[[VAL_0]] : i32 to %[[VAL_3]] : <i32>
// CHECK:           emitc.while {
// CHECK:             %[[VAL_4:.*]] = load %[[VAL_2]] : <i32>
// CHECK:             %[[VAL_5:.*]] = load %[[VAL_3]] : <i32>
// CHECK:             %[[VAL_6:.*]] = add %[[VAL_4]], %[[VAL_5]] : (i32, i32) -> i32
// CHECK:             %[[VAL_7:.*]] = cmp lt, %[[VAL_6]], %[[VAL_1]] : (i32, i32) -> i1
// CHECK:             yield %[[VAL_7]] : i1
// CHECK:           } do {
// CHECK:             %[[VAL_8:.*]] = load %[[VAL_2]] : <i32>
// CHECK:             %[[VAL_9:.*]] = load %[[VAL_3]] : <i32>
// CHECK:             %[[VAL_10:.*]] = call @payload_whileOp_two_results(%[[VAL_8]]) : (i32) -> i32
// CHECK:             %[[VAL_11:.*]] = call @payload_whileOp_two_results(%[[VAL_9]]) : (i32) -> i32
// CHECK:             assign %[[VAL_10]] : i32 to %[[VAL_2]] : <i32>
// CHECK:             assign %[[VAL_11]] : i32 to %[[VAL_3]] : <i32>
// CHECK:           }
// CHECK:           %[[VAL_12:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
// CHECK:           %[[VAL_13:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
// CHECK:           %[[VAL_14:.*]] = emitc.load %[[VAL_2]] : <i32>
// CHECK:           emitc.assign %[[VAL_14]] : i32 to %[[VAL_12]] : <i32>
// CHECK:           %[[VAL_15:.*]] = emitc.load %[[VAL_3]] : <i32>
// CHECK:           emitc.assign %[[VAL_15]] : i32 to %[[VAL_13]] : <i32>
// CHECK:           %[[VAL_16:.*]] = emitc.load %[[VAL_12]] : <i32>
// CHECK:           %[[VAL_17:.*]] = emitc.load %[[VAL_13]] : <i32>
// CHECK:           return %[[VAL_16]] : i32
// CHECK:         }

emitc.func @payload_doOp_two_results(%arg: i32) -> i32 {
  %result = emitc.add %arg, %arg : (i32, i32) -> i32
  return %result : i32
}

func.func @doOp_two_results() -> i32 {
  %init = emitc.literal "1.0" : i32
  %exit = emitc.literal "10.0" : i32

  %res1, %res2 = scf.while (%arg1_1 = %init, %arg1_2 = %init) : (i32, i32) -> (i32, i32) {
    %sum = emitc.add %arg1_1, %arg1_2 : (i32, i32) -> i32
    %condition = emitc.cmp lt, %sum, %exit : (i32, i32) -> i1
    scf.condition(%condition) %init, %arg1_2  : i32, i32
  } do {
  ^bb0(%arg2_1 : i32, %arg2_2 : i32):
    %next1 = emitc.call @payload_doOp_two_results(%arg2_1) : (i32) -> i32
    %next2 = emitc.call @payload_doOp_two_results(%arg2_2) : (i32) -> i32
    scf.yield %next1, %next2 : i32, i32
  }
  
  return %res1 : i32
}

// CHECK-LABEL:   emitc.func @payload_doOp_two_results(
// CHECK-SAME:      %[[ARG0:.*]]: i32) -> i32 {
// CHECK:           %[[VAL_0:.*]] = add %[[ARG0]], %[[ARG0]] : (i32, i32) -> i32
// CHECK:           return %[[VAL_0]] : i32
// CHECK:         }

// CHECK-LABEL:   func.func @doOp_two_results() -> i32 {
// CHECK:           %[[VAL_0:.*]] = emitc.literal "1.0" : i32
// CHECK:           %[[VAL_1:.*]] = emitc.literal "10.0" : i32
// CHECK:           %[[VAL_2:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
// CHECK:           emitc.assign %[[VAL_0]] : i32 to %[[VAL_2]] : <i32>
// CHECK:           %[[VAL_3:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
// CHECK:           emitc.assign %[[VAL_0]] : i32 to %[[VAL_3]] : <i32>
// CHECK:           %[[VAL_4:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i1>
// CHECK:           emitc.do {
// CHECK:             %[[VAL_5:.*]] = load %[[VAL_2]] : <i32>
// CHECK:             %[[VAL_6:.*]] = load %[[VAL_3]] : <i32>
// CHECK:             %[[VAL_7:.*]] = add %[[VAL_5]], %[[VAL_6]] : (i32, i32) -> i32
// CHECK:             %[[VAL_8:.*]] = cmp lt, %[[VAL_7]], %[[VAL_1]] : (i32, i32) -> i1
// CHECK:             assign %[[VAL_8]] : i1 to %[[VAL_4]] : <i1>
// CHECK:             if %[[VAL_8]] {
// CHECK:               %[[VAL_9:.*]] = call @payload_doOp_two_results(%[[VAL_0]]) : (i32) -> i32
// CHECK:               %[[VAL_10:.*]] = call @payload_doOp_two_results(%[[VAL_6]]) : (i32) -> i32
// CHECK:               assign %[[VAL_9]] : i32 to %[[VAL_2]] : <i32>
// CHECK:               assign %[[VAL_10]] : i32 to %[[VAL_3]] : <i32>
// CHECK:             }
// CHECK:           } while {
// CHECK:             %[[VAL_11:.*]] = load %[[VAL_4]] : <i1>
// CHECK:             yield %[[VAL_11]] : i1
// CHECK:           }
// CHECK:           %[[VAL_12:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
// CHECK:           %[[VAL_13:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
// CHECK:           %[[VAL_14:.*]] = emitc.load %[[VAL_2]] : <i32>
// CHECK:           emitc.assign %[[VAL_14]] : i32 to %[[VAL_12]] : <i32>
// CHECK:           %[[VAL_15:.*]] = emitc.load %[[VAL_3]] : <i32>
// CHECK:           emitc.assign %[[VAL_15]] : i32 to %[[VAL_13]] : <i32>
// CHECK:           %[[VAL_16:.*]] = emitc.load %[[VAL_12]] : <i32>
// CHECK:           %[[VAL_17:.*]] = emitc.load %[[VAL_13]] : <i32>
// CHECK:           return %[[VAL_16]] : i32
// CHECK:         }
