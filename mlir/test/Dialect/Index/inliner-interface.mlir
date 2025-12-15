// RUN: mlir-opt %s -inline | FileCheck %s

// CHECK-LABEL: @main
func.func @main(%arg0: i32) -> index {
  // CHECK-NOT: call
  // CHECK: index.castu
  %0 = call @f(%arg0) : (i32) -> index
  return %0 : index
}

// CHECK-LABEL: @f
func.func @f(%arg0: i32) -> index {
  %0 = index.castu %arg0 : i32 to index
  return %0 : index
}
