// RUN: mlir-opt %s | mlir-opt | FileCheck %s

func.func @member_call(%arg0 : !emitc.opaque<"MyClass">) {
  %0 = emitc.member_call_opaque %arg0 "method" () : !emitc.opaque<"MyClass">, () -> i32
  return
}
// CHECK-LABEL: func @member_call
// CHECK: emitc.member_call_opaque %arg0 "method"() : !emitc.opaque<"MyClass">, () -> i32

func.func @member_call_args(%arg0 : !emitc.opaque<"MyClass">, %arg1 : i32) {
  %0 = emitc.member_call_opaque %arg0 "method" (%arg1) : !emitc.opaque<"MyClass">, (i32) -> i32
  return
}
// CHECK-LABEL: func @member_call_args
// CHECK: emitc.member_call_opaque %arg0 "method"(%arg1) : !emitc.opaque<"MyClass">, (i32) -> i32

func.func @member_call_template_args(%arg0 : !emitc.opaque<"MyClass">) {
  %0 = emitc.member_call_opaque %arg0 "method" () {template_args = [i32]} : !emitc.opaque<"MyClass">, () -> i32
  return
}
// CHECK-LABEL: func @member_call_template_args
// CHECK: emitc.member_call_opaque %arg0 "method"() {template_args = [i32]} : !emitc.opaque<"MyClass">, () -> i32

func.func @member_call_reorder(%arg0 : !emitc.opaque<"MyClass">, %arg1 : i32, %arg2 : i32) {
  %0 = emitc.member_call_opaque %arg0 "method" (%arg1, %arg2) {args = [0 : index, 2 : index, 1 : index]} : !emitc.opaque<"MyClass">, (i32, i32) -> i32
  return
}
// CHECK-LABEL: func @member_call_reorder
// CHECK: emitc.member_call_opaque %arg0 "method"(%arg1, %arg2) {args = [0 : index, 2 : index, 1 : index]} : !emitc.opaque<"MyClass">, (i32, i32) -> i32
