// RUN: mlir-opt %s | FileCheck %s

// CHECK-LABEL: func @custom_region_names
func @custom_region_names() -> () {
  "test.polyfor"() ( {
  ^bb0(%arg0: index, %arg1: index, %arg2: index):
    "foo"() : () -> ()
  }) { arg_names = ["i", "j", "k"] } : () -> ()
  // CHECK: test.polyfor
  // CHECK-NEXT: ^bb{{.*}}(%i: index, %j: index, %k: index):
  return
}
