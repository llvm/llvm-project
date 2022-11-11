// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK: transform.sequence
// CHECK: ^{{.+}}(%{{.+}}: !pdl.operation):
transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  // CHECK: sequence %{{.+}} : !pdl.operation
  // CHECK: ^{{.+}}(%{{.+}}: !pdl.operation):
  sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
  }
}

// CHECK: transform.with_pdl_patterns
// CHECK: ^{{.+}}(%[[ARG:.+]]: !pdl.operation):
transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  // CHECK: sequence %[[ARG]] : !pdl.operation
  sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
  }
}

// CHECK: transform.sequence
// CHECK: ^{{.+}}(%[[ARG:.+]]: !pdl.operation):
transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  // CHECK: with_pdl_patterns %[[ARG]] : !pdl.operation
  with_pdl_patterns %arg0 : !pdl.operation {
  ^bb1(%arg1: !pdl.operation):
  }
}

// Using the same value multiple times without consuming it is fine.
// CHECK: transform.sequence
// CHECK: %[[V:.+]] = sequence %{{.*}} : !pdl.operation -> !pdl.operation
// CHECK: sequence %[[V]]
// CHECK: sequence %[[V]]
transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.sequence %arg0 : !pdl.operation -> !pdl.operation failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    yield %arg1 : !pdl.operation
  }
  transform.sequence %0 : !pdl.operation failures(propagate) {
  ^bb2(%arg2: !pdl.operation):
  }
  transform.sequence %0 : !pdl.operation failures(propagate) {
  ^bb3(%arg3: !pdl.operation):
  }
}

// CHECK: transform.sequence
// CHECK: foreach
transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  transform.foreach %arg0 : !pdl.operation {
  ^bb1(%arg1: !pdl.operation):
  }
}

// CHECK: transform.sequence
transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  // CHECK: cast %{{.*}} : !pdl.operation to !transform.any_op
  %0 = cast %arg0: !pdl.operation to !transform.any_op
  // CHECK: cast %{{.*}} : !transform.any_op to !transform.op<"builtin.module">
  %1 = cast %0: !transform.any_op to !transform.op<"builtin.module">
}

// CHECK: transform.sequence
// CHECK: print
// CHECK: print
// CHECK: print
// CHECK: print
transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  transform.print %arg0 : !pdl.operation
  transform.print
  transform.print %arg0 {name = "test"} : !pdl.operation
  transform.print {name = "test"}
}
