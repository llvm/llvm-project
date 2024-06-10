// RUN: mlir-opt -allow-unregistered-dialect -convert-scf-to-emitc %s | FileCheck %s

func.func @test_if(%arg0: i1, %arg1: f32) {
  scf.if %arg0 {
     %0 = emitc.call_opaque "func_const"(%arg1) : (f32) -> i32
  }
  return
}
// CHECK-LABEL: func.func @test_if(
// CHECK-SAME:                     %[[VAL_0:.*]]: i1,
// CHECK-SAME:                     %[[VAL_1:.*]]: f32) {
// CHECK-NEXT:    emitc.if %[[VAL_0]] {
// CHECK-NEXT:      %[[VAL_2:.*]] = emitc.call_opaque "func_const"(%[[VAL_1]]) : (f32) -> i32
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }


func.func @test_if_else(%arg0: i1, %arg1: f32) {
  scf.if %arg0 {
    %0 = emitc.call_opaque "func_true"(%arg1) : (f32) -> i32
  } else {
    %0 = emitc.call_opaque "func_false"(%arg1) : (f32) -> i32
  }
  return
}
// CHECK-LABEL: func.func @test_if_else(
// CHECK-SAME:                          %[[VAL_0:.*]]: i1,
// CHECK-SAME:                          %[[VAL_1:.*]]: f32) {
// CHECK-NEXT:    emitc.if %[[VAL_0]] {
// CHECK-NEXT:      %[[VAL_2:.*]] = emitc.call_opaque "func_true"(%[[VAL_1]]) : (f32) -> i32
// CHECK-NEXT:    } else {
// CHECK-NEXT:      %[[VAL_3:.*]] = emitc.call_opaque "func_false"(%[[VAL_1]]) : (f32) -> i32
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }


func.func @test_if_yield(%arg0: i1, %arg1: f32) {
  %0 = arith.constant 0 : i8
  %x, %y = scf.if %arg0 -> (i32, f64) {
    %1 = emitc.call_opaque "func_true_1"(%arg1) : (f32) -> i32
    %2 = emitc.call_opaque "func_true_2"(%arg1) : (f32) -> f64
    scf.yield %1, %2 : i32, f64
  } else {
    %1 = emitc.call_opaque "func_false_1"(%arg1) : (f32) -> i32
    %2 = emitc.call_opaque "func_false_2"(%arg1) : (f32) -> f64
    scf.yield %1, %2 : i32, f64
  }
  return
}
// CHECK-LABEL: func.func @test_if_yield(
// CHECK-SAME:                           %[[VAL_0:.*]]: i1,
// CHECK-SAME:                           %[[VAL_1:.*]]: f32) {
// CHECK-NEXT:    %[[VAL_2:.*]] = arith.constant 0 : i8
// CHECK-NEXT:    %[[VAL_3:.*]] = memref.alloca() : memref<i32>
// CHECK-NEXT:    %[[VAL_4:.*]] = memref.alloca() : memref<f64>
// CHECK-NEXT:    emitc.if %[[VAL_0]] {
// CHECK-NEXT:      %[[VAL_5:.*]] = emitc.call_opaque "func_true_1"(%[[VAL_1]]) : (f32) -> i32
// CHECK-NEXT:      %[[VAL_6:.*]] = emitc.call_opaque "func_true_2"(%[[VAL_1]]) : (f32) -> f64
// CHECK-NEXT:      memref.store %[[VAL_5]], %[[VAL_3]][] : memref<i32>
// CHECK-NEXT:      memref.store %[[VAL_6]], %[[VAL_4]][] : memref<f64>
// CHECK-NEXT:    } else {
// CHECK-NEXT:      %[[VAL_7:.*]] = emitc.call_opaque "func_false_1"(%[[VAL_1]]) : (f32) -> i32
// CHECK-NEXT:      %[[VAL_8:.*]] = emitc.call_opaque "func_false_2"(%[[VAL_1]]) : (f32) -> f64
// CHECK-NEXT:      memref.store %[[VAL_7]], %[[VAL_3]][] : memref<i32>
// CHECK-NEXT:      memref.store %[[VAL_8]], %[[VAL_4]][] : memref<f64>
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[VAL_9:.*]] = memref.load %[[VAL_3]][] : memref<i32>
// CHECK-NEXT:    %[[VAL_10:.*]] = memref.load %[[VAL_4]][] : memref<f64>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
