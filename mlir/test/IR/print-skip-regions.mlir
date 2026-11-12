// RUN: mlir-opt --no-implicit-module --mlir-print-skip-regions \
// RUN:   --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @foo(%{{.+}}: i32, %{{.+}}: i32, %{{.+}}: i32) -> i32 {...}
// CHECK-NOT:     return
func.func @foo(%arg0: i32, %arg1: i32, %arg3: i32) -> i32 {
  return %arg0: i32
}

// -----

// CHECK: module {...}
// CHECK-NOT: func.func
module {
  func.func @foo(%arg0: i32, %arg1: i32, %arg3: i32) -> i32 {
    return %arg0: i32
  }
}
