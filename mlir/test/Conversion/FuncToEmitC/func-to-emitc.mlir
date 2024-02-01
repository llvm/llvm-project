// RUN: mlir-opt -split-input-file -convert-func-to-emitc %s | FileCheck %s

// CHECK-LABEL: emitc.func @foo()
// CHECK-NEXT: emitc.return
func.func @foo() {
  return
}

// -----

// CHECK-LABEL: emitc.func private @foo() attributes {specifiers = ["static"]}
// CHECK-NEXT: emitc.return
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
// CHECK-NEXT: emitc.return %arg0 : i32
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
// CHECK-NEXT: emitc.return %arg0 : i32
func.func private @return_i32(%arg0: i32) -> i32 {
  return %arg0 : i32
}

// CHECK-LABEL: emitc.func @call(%arg0: i32) -> i32
// CHECK-NEXT: %0 = emitc.call @return_i32(%arg0) : (i32) -> i32
// CHECK-NEXT: emitc.return %0 : i32
func.func @call(%arg0: i32) -> i32 {
  %0 = call @return_i32(%arg0) : (i32) -> (i32)
  return %0 : i32
}
