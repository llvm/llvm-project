// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// RUN: mlir-opt %s -canonicalize | FileCheck %s

// CHECK: emitc.include <"test.h">
// CHECK: emitc.include "test.h"
emitc.include <"test.h">
emitc.include "test.h"

// CHECK-LABEL: func @f(%{{.*}}: i32, %{{.*}}: !emitc.opaque<"int32_t">) {
func.func @f(%arg0: i32, %f: !emitc.opaque<"int32_t">) {
  %1 = "emitc.call"() {callee = "blah"} : () -> i64
  emitc.call "foo" (%1) {args = [
    0 : index, dense<[0, 1]> : tensor<2xi32>, 0 : index
  ]} : (i64) -> ()
  return
}

func.func @cast(%arg0: i32) {
  %1 = emitc.cast %arg0: i32 to f32
  return
}

func.func @c() {
  %1 = "emitc.constant"(){value = 42 : i32} : () -> i32
  return
}

func.func @a(%arg0: i32, %arg1: i32) {
  %1 = "emitc.apply"(%arg0) {applicableOperator = "&"} : (i32) -> !emitc.ptr<i32>
  %2 = emitc.apply "&"(%arg1) : (i32) -> !emitc.ptr<i32>
  return
}

func.func @add_int(%arg0: i32, %arg1: i32) {
  %1 = "emitc.add" (%arg0, %arg1) : (i32, i32) -> i32
  return
}

func.func @add_pointer(%arg0: !emitc.ptr<f32>, %arg1: i32, %arg2: !emitc.opaque<"unsigned int">) {
  %1 = "emitc.add" (%arg0, %arg1) : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
  %2 = "emitc.add" (%arg0, %arg2) : (!emitc.ptr<f32>, !emitc.opaque<"unsigned int">) -> !emitc.ptr<f32>
  return
}

func.func @div_int(%arg0: i32, %arg1: i32) {
  %1 = "emitc.div" (%arg0, %arg1) : (i32, i32) -> i32
  return
}

func.func @div_float(%arg0: f32, %arg1: f32) {
  %1 = "emitc.div" (%arg0, %arg1) : (f32, f32) -> f32
  return
}

func.func @mul_int(%arg0: i32, %arg1: i32) {
  %1 = "emitc.mul" (%arg0, %arg1) : (i32, i32) -> i32
  return
}

func.func @mul_float(%arg0: f32, %arg1: f32) {
  %1 = "emitc.mul" (%arg0, %arg1) : (f32, f32) -> f32
  return
}

func.func @rem(%arg0: i32, %arg1: i32) {
  %1 = "emitc.rem" (%arg0, %arg1) : (i32, i32) -> i32
  return
}

func.func @sub_int(%arg0: i32, %arg1: i32) {
  %1 = "emitc.sub" (%arg0, %arg1) : (i32, i32) -> i32
  return
}

func.func @sub_pointer(%arg0: !emitc.ptr<f32>, %arg1: i32, %arg2: !emitc.opaque<"unsigned int">, %arg3: !emitc.ptr<f32>) {
  %1 = "emitc.sub" (%arg0, %arg1) : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
  %2 = "emitc.sub" (%arg0, %arg2) : (!emitc.ptr<f32>, !emitc.opaque<"unsigned int">) -> !emitc.ptr<f32>
  %3 = "emitc.sub" (%arg0, %arg3) : (!emitc.ptr<f32>, !emitc.ptr<f32>) -> !emitc.opaque<"ptrdiff_t">
  %4 = "emitc.sub" (%arg0, %arg3) : (!emitc.ptr<f32>, !emitc.ptr<f32>) -> i32
  return
}
