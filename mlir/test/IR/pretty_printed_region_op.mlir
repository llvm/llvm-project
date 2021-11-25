// RUN: mlir-opt -allow-unregistered-dialect -split-input-file  %s | FileCheck %s --check-prefixes=CHECK-CUSTOM,CHECK
// RUN: mlir-opt -allow-unregistered-dialect -mlir-print-op-generic -split-input-file   %s | FileCheck %s --check-prefixes=CHECK,CHECK-GENERIC

// -----

func @pretty_printed_region_op(%arg0 : f32, %arg1 : f32) -> (f32) {
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

func @pretty_printed_region_op(%arg0 : f32, %arg1 : f32) -> (f32) {
// CHECK-CUSTOM:  test.pretty_printed_region %arg1, %arg0
// CHECK-GENERIC: "test.pretty_printed_region"(%arg1, %arg0)
// CHECK:          ^bb0(%arg[[x:[0-9]+]]: f32, %arg[[y:[0-9]+]]: f32):
// CHECK:            %[[RES:.*]] = "non.special.op"(%arg[[x]], %arg[[y]]) : (f32, f32) -> f32
// CHECK:            "test.return"(%[[RES]]) : (f32) -> ()
// CHECK:          : (f32, f32) -> f32

  %0 = test.pretty_printed_region %arg1, %arg0 ( {
    ^bb0(%arg2: f32, %arg3: f32):
      %1 = "non.special.op"(%arg2, %arg3) : (f32, f32) -> f32
      "test.return"(%1) : (f32) -> ()
    }) : (f32, f32) -> f32
    return %0 : f32
}

