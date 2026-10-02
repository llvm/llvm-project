// RUN: mlir-opt %s -canonicalize="test-convergence" | FileCheck %s

//===----------------------------------------------------------------------===//
// UnrealizedConversionCastOp
//===----------------------------------------------------------------------===//

// Test folding conversion casts feeding into other casts.
// CHECK-LABEL: func @multiple_conversion_casts
// CHECK-SAME: %[[ARG0:.*]]: i32, %[[ARG1:.*]]:
func.func @multiple_conversion_casts(%arg0: i32, %arg1: i32) -> (i32, i32) {
  // CHECK-NOT: unrealized_conversion_cast
  // CHECK: return %[[ARG0]], %[[ARG1]]
  %inputs:2 = builtin.unrealized_conversion_cast %arg0, %arg1 : i32, i32 to i64, i64
  %outputs:2 = builtin.unrealized_conversion_cast %inputs#0, %inputs#1 : i64, i64 to i32, i32
  return %outputs#0, %outputs#1 : i32, i32
}

// CHECK-LABEL: func @multiple_conversion_casts
func.func @multiple_conversion_casts_failure(%arg0: i32, %arg1: i32, %arg2: i64) -> (i32, i32) {
  // CHECK: unrealized_conversion_cast
  // CHECK: unrealized_conversion_cast
  %inputs:2 = builtin.unrealized_conversion_cast %arg0, %arg1 : i32, i32 to i64, i64
  %outputs:2 = builtin.unrealized_conversion_cast %arg2, %inputs#1 : i64, i64 to i32, i32
  return %outputs#0, %outputs#1 : i32, i32
}

// In a graph region, a cast can forward its own result. The fold keeps that
// result, so the cast stays.
// CHECK-LABEL: func @graph_region_self_forward
//       CHECK:   test.graph_region
//       CHECK:     %[[CAST:.+]] = builtin.unrealized_conversion_cast %[[CAST]] : i32 to i32
func.func @graph_region_self_forward() {
  test.graph_region {
    %0 = builtin.unrealized_conversion_cast %0 : i32 to i32
    "test.use"(%0) : (i32) -> ()
  }
  return
}

// In a graph region, a forwarded operand can be another result of the same
// cast. The fold also replaces that result, so it must not apply.
// CHECK-LABEL: func @graph_region_chain
//       CHECK:   test.graph_region
//       CHECK:     %[[CAST:.+]]:2 = builtin.unrealized_conversion_cast %[[CAST]]#1, %{{.+}} : i32, i32 to i32, i32
func.func @graph_region_chain(%x: i32) {
  test.graph_region {
    %0:2 = builtin.unrealized_conversion_cast %0#1, %x : i32, i32 to i32, i32
    "test.use"(%0#0, %0#1) : (i32, i32) -> ()
  }
  return
}

// Same as above, but the forwarded operands form a cycle.
// CHECK-LABEL: func @graph_region_cycle
//       CHECK:   test.graph_region
//       CHECK:     %[[CAST:.+]]:2 = builtin.unrealized_conversion_cast %[[CAST]]#1, %[[CAST]]#0 : i32, i32 to i32, i32
func.func @graph_region_cycle() {
  test.graph_region {
    %0:2 = builtin.unrealized_conversion_cast %0#1, %0#0 : i32, i32 to i32, i32
    "test.use"(%0#0, %0#1) : (i32, i32) -> ()
  }
  return
}

// Same as above, but the second operand forwards the first result.
// CHECK-LABEL: func @graph_region_backward_chain
//       CHECK:   test.graph_region
//       CHECK:     %[[CAST:.+]]:2 = builtin.unrealized_conversion_cast %{{.+}}, %[[CAST]]#0 : i32, i32 to i32, i32
func.func @graph_region_backward_chain(%x: i32) {
  test.graph_region {
    %0:2 = builtin.unrealized_conversion_cast %x, %0#0 : i32, i32 to i32, i32
    "test.use"(%0#0, %0#1) : (i32, i32) -> ()
  }
  return
}
