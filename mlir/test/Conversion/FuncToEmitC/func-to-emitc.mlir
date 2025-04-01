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
