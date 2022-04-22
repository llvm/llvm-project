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

// CHECK: transform.with_pdl_patterns
// CHECK: ^{{.+}}(%[[ARG:.+]]: !pdl.operation):
transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  // CHECK: sequence %[[ARG]]
  sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):
  }
}

// CHECK: transform.sequence
// CHECK: ^{{.+}}(%[[ARG:.+]]: !pdl.operation):
transform.sequence {
^bb0(%arg0: !pdl.operation):
  // CHECK: with_pdl_patterns %[[ARG]]
  with_pdl_patterns %arg0 {
  ^bb1(%arg1: !pdl.operation):
  }
}
