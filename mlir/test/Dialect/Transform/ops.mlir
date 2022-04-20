// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK: transform.sequence
// CHECK: ^{{.+}}(%{{.+}}: !pdl.operation):
transform.sequence {
^bb0(%arg0: !pdl.operation):
  // CHECK: sequence %{{.+}}
  // CHECK: ^{{.+}}(%{{.+}}: !pdl.operation):
  sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):
  }
}
