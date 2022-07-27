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

// Using the same value multiple times without consuming it is fine.
// CHECK: transform.sequence
// CHECK: %[[V:.+]] = sequence
// CHECK: sequence %[[V]]
// CHECK: sequence %[[V]]
transform.sequence {
^bb0(%arg0: !pdl.operation):
  %0 = transform.sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):
    yield %arg1 : !pdl.operation
  } : !pdl.operation
  transform.sequence %0 {
  ^bb2(%arg2: !pdl.operation):
  }
  transform.sequence %0 {
  ^bb3(%arg3: !pdl.operation):
  }
}

// CHECK: transform.sequence
// CHECK: foreach
transform.sequence {
^bb0(%arg0: !pdl.operation):
  transform.foreach %arg0 {
  ^bb1(%arg1: !pdl.operation):
  }
}
