// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s

// CHECK: int32_t lvalue_variables(
emitc.func @lvalue_variables(%v1: i32, %v2: i32) -> i32 {
  %val = emitc.mul %v1, %v2 : (i32, i32) -> i32
  %variable = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.lvalue<i32> // alloc effect
  emitc.assign %val : i32 to %variable : !emitc.lvalue<i32> // write effect
  %addr = emitc.apply "&"(%variable) : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>
  emitc.call @zero (%addr) : (!emitc.ptr<i32>) -> ()
  %updated_val = emitc.lvalue_load %variable : !emitc.lvalue<i32> // read effect, (noop in emitter?)
  %neg_one = "emitc.constant"() {value = -1 : i32} : () -> i32
  emitc.assign %neg_one : i32 to %variable : !emitc.lvalue<i32> // invalidates %updated_val
  emitc.return %updated_val : i32
  // dealloc effect through automatic allocation scope
}

emitc.func @zero(%arg0: !emitc.ptr<i32>) attributes {specifiers = ["extern"]}