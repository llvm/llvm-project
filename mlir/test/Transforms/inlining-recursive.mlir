// RUN: mlir-opt %s -inline='default-pipeline=' | FileCheck %s
// RUN: mlir-opt %s --mlir-disable-threading -inline='default-pipeline=' | FileCheck %s
// RUN: mlir-opt %s -inline | FileCheck %s --check-prefix=DEFAULT

// CHECK-LABEL: func.func @foo0
func.func @foo0(%arg0 : i32) -> i32 {
  // CHECK: call @foo1
  // CHECK: }
  %0 = arith.constant 0 : i32
  %1 = arith.cmpi eq, %arg0, %0 : i32
  cf.cond_br %1, ^exit, ^tail
^exit:
  return %0 : i32
^tail:
  %3 = call @foo1(%arg0) : (i32) -> i32
  return %3 : i32
}

// CHECK-LABEL: func.func @foo1
func.func @foo1(%arg0 : i32) -> i32 {
  // CHECK:    call @foo0
  %0 = arith.constant 1 : i32
  %1 = arith.subi %arg0, %0 : i32
  %2 = call @foo0(%1) : (i32) -> i32
  return %2 : i32
}

// Verify that recursive expansion is bounded across inliner iterations.
// CHECK-LABEL: func.func @caller
// CHECK-NEXT: %{{.*}} = arith.constant 1 : i32
// CHECK-NEXT: %{{.*}} = arith.addi
// CHECK-NEXT: %{{.*}} = arith.constant 2 : i32
// CHECK-NEXT: %{{.*}} = arith.addi
// CHECK-NEXT: %{{.*}} = call @cycle_a
// CHECK-NEXT: %{{.*}} = constant @leaf
// CHECK-NEXT: %{{.*}} = call_indirect
// CHECK-NEXT: %{{.*}} = constant @cycle_a
// CHECK-NEXT: %{{.*}} = call_indirect
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @caller(%arg0 : i32) -> i32 {
  %0 = call @cycle_a(%arg0) : (i32) -> i32
  %1 = call @wrapper(%0) : (i32) -> i32
  %2 = call @same_edge_wrapper(%1) : (i32) -> i32
  return %2 : i32
}

// DEFAULT-LABEL: func.func @caller
// DEFAULT-NEXT: %{{.*}} = arith.constant 6 : i32
// DEFAULT-NEXT: %{{.*}} = arith.constant 4 : i32
// DEFAULT-NEXT: %{{.*}} = arith.constant 3 : i32
// DEFAULT-NEXT: %{{.*}} = arith.addi
// DEFAULT-NEXT: %{{.*}} = call @cycle_a
// DEFAULT-NEXT: %{{.*}} = arith.muli
// DEFAULT-NEXT: %{{.*}} = arith.addi
// DEFAULT-NEXT: %{{.*}} = call @cycle_a
// DEFAULT-NEXT: return
// DEFAULT-NEXT: }

// CHECK-LABEL: func.func @cycle_a
// DEFAULT-LABEL: func.func @cycle_a
func.func @cycle_a(%arg0 : i32) -> i32 {
  %c1 = arith.constant 1 : i32
  %0 = arith.addi %arg0, %c1 : i32
  %1 = call @cycle_b(%0) : (i32) -> i32
  return %1 : i32
}

func.func @cycle_b(%arg0 : i32) -> i32 {
  %c2 = arith.constant 2 : i32
  %0 = arith.addi %arg0, %c2 : i32
  %1 = call @cycle_a(%0) : (i32) -> i32
  return %1 : i32
}

func.func @wrapper(%arg0 : i32) -> i32 {
  %fn = constant @leaf : (i32) -> i32
  %0 = call_indirect %fn(%arg0) : (i32) -> i32
  return %0 : i32
}

func.func @same_edge_wrapper(%arg0 : i32) -> i32 {
  %fn = constant @cycle_a : (i32) -> i32
  %0 = call_indirect %fn(%arg0) : (i32) -> i32
  return %0 : i32
}

func.func @leaf(%arg0 : i32) -> i32 {
  %c4 = arith.constant 4 : i32
  %0 = arith.muli %arg0, %c4 : i32
  return %0 : i32
}

// CHECK-LABEL: func.func @two_calls
// CHECK-NEXT: %{{.*}} = arith.constant 1 : i32
// CHECK-NEXT: %{{.*}} = arith.addi
// CHECK-NEXT: %{{.*}} = arith.constant 2 : i32
// CHECK-NEXT: %{{.*}} = arith.addi
// CHECK-NEXT: %{{.*}} = call @cycle_a
// CHECK-NEXT: %{{.*}} = arith.constant 1 : i32
// CHECK-NEXT: %{{.*}} = arith.addi
// CHECK-NEXT: %{{.*}} = arith.constant 2 : i32
// CHECK-NEXT: %{{.*}} = arith.addi
// CHECK-NEXT: %{{.*}} = call @cycle_a
// CHECK-NEXT: return
// CHECK-NEXT: }
// DEFAULT-LABEL: func.func @two_calls
// DEFAULT-NEXT: %{{.*}} = arith.constant 3 : i32
// DEFAULT-NEXT: %{{.*}} = arith.addi
// DEFAULT-NEXT: %{{.*}} = call @cycle_a
// DEFAULT-NEXT: %{{.*}} = arith.addi
// DEFAULT-NEXT: %{{.*}} = call @cycle_a
// DEFAULT-NEXT: return
// DEFAULT-NEXT: }
func.func @two_calls(%arg0 : i32) -> i32 {
  %0 = call @cycle_a(%arg0) : (i32) -> i32
  %1 = call @cycle_a(%0) : (i32) -> i32
  return %1 : i32
}
