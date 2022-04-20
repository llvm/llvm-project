// RUN: mlir-opt -allow-unregistered-dialect -split-input-file %s | FileCheck %s --check-prefixes=CHECK-CUSTOM,CHECK
// RUN: mlir-opt -allow-unregistered-dialect -mlir-print-op-generic -split-input-file   %s | FileCheck %s --check-prefixes=CHECK,CHECK-GENERIC
// RUN: mlir-opt -allow-unregistered-dialect -split-input-file --mlir-print-op-generic --mlir-print-debuginfo -mlir-print-local-scope  %s | FileCheck %s --check-prefixes=CHECK-LOCATION

// -----

func.func @pretty_printed_region_op(%arg0 : f32, %arg1 : f32) -> (f32) {
// CHECK-CUSTOM:  test.pretty_printed_region %arg1, %arg0 start special.op end : (f32, f32) -> f32
// CHECK-GENERIC: "test.pretty_printed_region"(%arg1, %arg0)
// CHECK-GENERIC:   ^bb0(%arg[[x:[0-9]+]]: f32, %arg[[y:[0-9]+]]: f32
// CHECK-GENERIC:     %[[RES:.*]] = "special.op"(%arg[[x]], %arg[[y]]) : (f32, f32) -> f32
// CHECK-GENERIC:     "test.return"(%[[RES]]) : (f32) -> ()
// CHECK-GENERIC:  : (f32, f32) -> f32

  %res = test.pretty_printed_region %arg1, %arg0 start special.op end : (f32, f32) -> (f32) loc("some_NameLoc")
  return %res : f32
}

// -----

func.func @pretty_printed_region_op(%arg0 : f32, %arg1 : f32) -> (f32) {
// CHECK-CUSTOM:  test.pretty_printed_region %arg1, %arg0
// CHECK-GENERIC: "test.pretty_printed_region"(%arg1, %arg0)
// CHECK:          ^bb0(%arg[[x:[0-9]+]]: f32, %arg[[y:[0-9]+]]: f32):
// CHECK:            %[[RES:.*]] = "non.special.op"(%arg[[x]], %arg[[y]]) : (f32, f32) -> f32
// CHECK:            "test.return"(%[[RES]]) : (f32) -> ()
// CHECK:          : (f32, f32) -> f32

  %0 = test.pretty_printed_region %arg1, %arg0 ({
    ^bb0(%arg2: f32, %arg3: f32):
      %1 = "non.special.op"(%arg2, %arg3) : (f32, f32) -> f32
      "test.return"(%1) : (f32) -> ()
    }) : (f32, f32) -> f32
    return %0 : f32
}

// -----

func.func @pretty_printed_region_op_deferred_loc(%arg0 : f32, %arg1 : f32) -> (f32) {
// CHECK-LOCATION: "test.pretty_printed_region"(%arg1, %arg0)
// CHECK-LOCATION:   ^bb0(%arg[[x:[0-9]+]]: f32 loc("foo"), %arg[[y:[0-9]+]]: f32 loc("foo")
// CHECK-LOCATION:     %[[RES:.*]] = "special.op"(%arg[[x]], %arg[[y]]) : (f32, f32) -> f32
// CHECK-LOCATION:     "test.return"(%[[RES]]) : (f32) -> ()
// CHECK-LOCATION:  : (f32, f32) -> f32

  %res = test.pretty_printed_region %arg1, %arg0 start special.op end : (f32, f32) -> (f32) loc("foo")
  return %res : f32
}

// -----

// This tests the behavior of custom block names:
// operations like `test.block_names` can define custom names for blocks in
// nested regions.
// CHECK-CUSTOM-LABEL: func @block_names
func.func @block_names(%bool : i1) {
  // CHECK: test.block_names
  test.block_names {
    // CHECK-CUSTOM: br ^foo1
    // CHECK-GENERIC: cf.br{{.*}}^bb1
    cf.br ^foo1
  // CHECK-CUSTOM: ^foo1:
  // CHECK-GENERIC: ^bb1:
  ^foo1:
    // CHECK-CUSTOM: br ^foo2
    // CHECK-GENERIC: cf.br{{.*}}^bb2
    cf.br ^foo2
  // CHECK-CUSTOM: ^foo2:
  // CHECK-GENERIC: ^bb2:
  ^foo2:
     "test.return"() : () -> ()
  }
  return
}
