// RUN: mlir-opt -allow-unregistered-dialect -convert-scf-to-emitc %s | FileCheck %s

// CHECK-LABEL:   func.func @switch_no_result(
// CHECK-SAME:                                %[[ARG_0:.*]]: index) {
// CHECK:           %[[VAL_0:.*]] = builtin.unrealized_conversion_cast %[[ARG_0]] : index to !emitc.size_t
// CHECK:           emitc.switch %[[VAL_0]]
// CHECK:           case 2 {
// CHECK:             %[[VAL_1:.*]] = arith.constant 10 : i32
// CHECK:             emitc.yield
// CHECK:           }
// CHECK:           case 5 {
// CHECK:             %[[VAL_2:.*]] = arith.constant 20 : i32
// CHECK:             emitc.yield
// CHECK:           }
// CHECK:           default {
// CHECK:             %[[VAL_3:.*]] = arith.constant 30 : i32
// CHECK:           }
// CHECK:           return
// CHECK:         }
func.func @switch_no_result(%arg0 : index) {
    scf.index_switch %arg0
    case 2 {
      %1 = arith.constant 10 : i32
      scf.yield
    }
    case 5 {
      %2 = arith.constant 20 : i32
      scf.yield
    }
    default {
      %3 = arith.constant 30 : i32
    }
  return
}

// CHECK-LABEL:   func.func @switch_one_result(
// CHECK-SAME:                                 %[[ARG_0:.*]]: index) {
// CHECK:           %[[VAL_0:.*]] = builtin.unrealized_conversion_cast %[[ARG_0]] : index to !emitc.size_t
// CHECK:           %[[VAL_1:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
// CHECK:           emitc.switch %[[VAL_0]]
// CHECK:           case 2 {
// CHECK:             %[[VAL_2:.*]] = arith.constant 10 : i32
// CHECK:             emitc.assign %[[VAL_2]] : i32 to %[[VAL_1]] : <i32>
// CHECK:             emitc.yield
// CHECK:           }
// CHECK:           case 5 {
// CHECK:             %[[VAL_3:.*]] = arith.constant 20 : i32
// CHECK:             emitc.assign %[[VAL_3]] : i32 to %[[VAL_1]] : <i32>
// CHECK:             emitc.yield
// CHECK:           }
// CHECK:           default {
// CHECK:             %[[VAL_4:.*]] = arith.constant 30 : i32
// CHECK:             emitc.assign %[[VAL_4]] : i32 to %[[VAL_1]] : <i32>
// CHECK:           }
// CHECK:           return
// CHECK:         }
func.func @switch_one_result(%arg0 : index) {
    %0 = scf.index_switch %arg0 -> i32
    case 2 {
      %1 = arith.constant 10 : i32
      scf.yield %1 : i32
    }
    case 5 {
      %2 = arith.constant 20 : i32
      scf.yield %2 : i32
    }
    default {
      %3 = arith.constant 30 : i32
      scf.yield %3 : i32
    }
  return
}

// CHECK-LABEL:   func.func @switch_two_results(
// CHECK-SAME:                                  %[[ARG_0:.*]]: index) -> (i32, f32) {
// CHECK:           %[[VAL_0:.*]] = builtin.unrealized_conversion_cast %[[ARG_0]] : index to !emitc.size_t
// CHECK:           %[[VAL_1:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
// CHECK:           %[[VAL_2:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<f32>
// CHECK:           emitc.switch %[[VAL_0]]
// CHECK:           case 2 {
// CHECK:             %[[VAL_3:.*]] = arith.constant 10 : i32
// CHECK:             %[[VAL_4:.*]] = arith.constant 1.200000e+00 : f32
// CHECK:             emitc.assign %[[VAL_3]] : i32 to %[[VAL_1]] : <i32>
// CHECK:             emitc.assign %[[VAL_4]] : f32 to %[[VAL_2]] : <f32>
// CHECK:             emitc.yield
// CHECK:           }
// CHECK:           case 5 {
// CHECK:             %[[VAL_5:.*]] = arith.constant 20 : i32
// CHECK:             %[[VAL_6:.*]] = arith.constant 2.400000e+00 : f32
// CHECK:             emitc.assign %[[VAL_5]] : i32 to %[[VAL_1]] : <i32>
// CHECK:             emitc.assign %[[VAL_6]] : f32 to %[[VAL_2]] : <f32>
// CHECK:             emitc.yield
// CHECK:           }
// CHECK:           default {
// CHECK:             %[[VAL_7:.*]] = arith.constant 30 : i32
// CHECK:             %[[VAL_8:.*]] = arith.constant 3.600000e+00 : f32
// CHECK:             emitc.assign %[[VAL_7]] : i32 to %[[VAL_1]] : <i32>
// CHECK:             emitc.assign %[[VAL_8]] : f32 to %[[VAL_2]] : <f32>
// CHECK:           }
// CHECK:           %[[RES_1:.*]] = emitc.load %[[VAL_1]] : <i32>
// CHECK:           %[[RES_2:.*]] = emitc.load %[[VAL_2]] : <f32>
// CHECK:           return %[[RES_1]], %[[RES_2]] : i32, f32
// CHECK:         }
func.func @switch_two_results(%arg0 : index) -> (i32, f32) {
    %0, %1 = scf.index_switch %arg0 -> i32, f32
    case 2 {
      %2 = arith.constant 10 : i32
      %3 = arith.constant 1.2 : f32
      scf.yield %2, %3 : i32, f32
    }
    case 5 {
      %4 = arith.constant 20 : i32
      %5 = arith.constant 2.4 : f32
      scf.yield %4, %5 : i32, f32
    }
    default {
      %6 = arith.constant 30 : i32
      %7 = arith.constant 3.6 : f32
      scf.yield %6, %7 : i32, f32
    }
    return %0, %1 : i32, f32
}
