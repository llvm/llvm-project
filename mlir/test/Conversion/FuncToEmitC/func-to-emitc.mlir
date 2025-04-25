// RUN: mlir-opt -split-input-file -convert-func-to-emitc %s | FileCheck %s

// CHECK-LABEL: emitc.func @foo()
// CHECK-NEXT: return
func.func @foo() {
  return
}

// -----

// CHECK-LABEL: emitc.func private @foo() attributes {specifiers = ["static"]}
// CHECK-NEXT: return
func.func private @foo() {
  return
}

// -----

// CHECK-LABEL: emitc.func @foo(%arg0: i32)
func.func @foo(%arg0: i32) {
  emitc.call_opaque "bar"(%arg0) : (i32) -> ()
  return
}

// -----

// CHECK-LABEL: emitc.func @foo(%arg0: i32) -> i32
// CHECK-NEXT: return %arg0 : i32
func.func @foo(%arg0: i32) -> i32 {
  return %arg0 : i32
}

// -----

// CHECK-LABEL: emitc.func @foo(%arg0: i32, %arg1: i32) -> i32
func.func @foo(%arg0: i32, %arg1: i32) -> i32 {
  %0 = "emitc.add" (%arg0, %arg1) : (i32, i32) -> i32
  return %0 : i32
}

// -----

// CHECK-LABEL: emitc.func private @return_i32(%arg0: i32) -> i32 attributes {specifiers = ["static"]}
// CHECK-NEXT: return %arg0 : i32
func.func private @return_i32(%arg0: i32) -> i32 {
  return %arg0 : i32
}

// CHECK-LABEL: emitc.func @call(%arg0: i32) -> i32
// CHECK-NEXT: %0 = call @return_i32(%arg0) : (i32) -> i32
// CHECK-NEXT: return %0 : i32
func.func @call(%arg0: i32) -> i32 {
  %0 = call @return_i32(%arg0) : (i32) -> (i32)
  return %0 : i32
}

// -----

// CHECK-LABEL: emitc.func private @return_i32(i32) -> i32 attributes {specifiers = ["extern"]}
func.func private @return_i32(%arg0: i32) -> i32

// -----

// CHECK-LABEL: emitc.func private @return_void() attributes {specifiers = ["static"]}
// CHECK-NEXT: return
func.func private @return_void() {
  return
}

// CHECK-LABEL: emitc.func @call()
// CHECK-NEXT: call @return_void() : () -> ()
// CHECK-NEXT: return
func.func @call() {
  call @return_void() : () -> ()
  return
}

// -----

// CHECK-LABEL: emitc.func @use_index
// CHECK-SAME: (%[[Arg0:.*]]: !emitc.size_t) -> !emitc.size_t
// CHECK: return %[[Arg0]] : !emitc.size_t
func.func @use_index(%arg0: index) -> index {
  return %arg0 : index
}

// -----

// CHECK-LABEL: emitc.func private @prototype_index(!emitc.size_t) -> !emitc.size_t attributes {specifiers = ["extern"]}
func.func private @prototype_index(%arg0: index) -> index

// CHECK-LABEL: emitc.func @call(%arg0: !emitc.size_t) -> !emitc.size_t
// CHECK-NEXT: %0 = call @prototype_index(%arg0) : (!emitc.size_t) -> !emitc.size_t
// CHECK-NEXT: return %0 : !emitc.size_t
func.func @call(%arg0: index) -> index {
  %0 = call @prototype_index(%arg0) : (index) -> (index)
  return %0 : index
}

// -----

// CHECK-LABEL: emitc.func @index_args_only(%arg0: !emitc.size_t) -> f32
func.func @index_args_only(%i: index) -> f32 {
  %0 = arith.constant 0.0 : f32
  return %0 : f32
}
