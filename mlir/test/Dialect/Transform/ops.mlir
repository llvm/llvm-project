// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK: transform.sequence
// CHECK: ^{{.+}}(%{{.+}}: !transform.any_op):
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  // CHECK: sequence %{{.+}} : !transform.any_op
  // CHECK: ^{{.+}}(%{{.+}}: !transform.any_op):
  sequence %arg0 : !transform.any_op failures(propagate) {
  ^bb1(%arg1: !transform.any_op):
  }
}

// CHECK: transform.with_pdl_patterns
// CHECK: ^{{.+}}(%[[ARG:.+]]: !transform.any_op):
transform.with_pdl_patterns {
^bb0(%arg0: !transform.any_op):
  // CHECK: sequence %[[ARG]] : !transform.any_op
  sequence %arg0 : !transform.any_op failures(propagate) {
  ^bb1(%arg1: !transform.any_op):
  }
}

// Using the same value multiple times without consuming it is fine.
// CHECK: transform.sequence
// CHECK: %[[V:.+]] = sequence %{{.*}} : !transform.any_op -> !transform.any_op
// CHECK: sequence %[[V]]
// CHECK: sequence %[[V]]
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %0 = transform.sequence %arg0 : !transform.any_op -> !transform.any_op failures(propagate) {
  ^bb1(%arg1: !transform.any_op):
    yield %arg1 : !transform.any_op
  }
  transform.sequence %0 : !transform.any_op failures(propagate) {
  ^bb2(%arg2: !transform.any_op):
  }
  transform.sequence %0 : !transform.any_op failures(propagate) {
  ^bb3(%arg3: !transform.any_op):
  }
}

// CHECK: transform.sequence failures(propagate)
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op, %arg1: !transform.any_op, %arg2: !transform.any_op):
  // CHECK: sequence %{{.*}}, %{{.*}}, %{{.*}} : (!transform.any_op, !transform.any_op, !transform.any_op) failures(propagate)
  transform.sequence %arg0, %arg1, %arg2 : !transform.any_op, !transform.any_op, !transform.any_op failures(propagate) {
  ^bb0(%arg3: !transform.any_op, %arg4: !transform.any_op, %arg5: !transform.any_op):
  }
}

// CHECK: transform.sequence failures(propagate)
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op, %arg1: !transform.any_op, %arg2: !transform.any_op):
  // CHECK: sequence %{{.*}}, %{{.*}}, %{{.*}} : (!transform.any_op, !transform.any_op, !transform.any_op) failures(propagate)
  transform.sequence %arg0, %arg1, %arg2 : (!transform.any_op, !transform.any_op, !transform.any_op) failures(propagate) {
  ^bb0(%arg3: !transform.any_op, %arg4: !transform.any_op, %arg5: !transform.any_op):
  }
}

// CHECK: transform.sequence failures(propagate)
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op, %arg1: !transform.any_op, %arg2: !transform.any_op):
  // CHECK: sequence %{{.*}}, %{{.*}}, %{{.*}} : (!transform.any_op, !transform.any_op, !transform.any_op) failures(propagate)
  transform.sequence %arg0, %arg1, %arg2 : (!transform.any_op, !transform.any_op, !transform.any_op) failures(propagate) {
  ^bb0(%arg3: !transform.any_op, %arg4: !transform.any_op, %arg5: !transform.any_op):
  }
}

// CHECK: transform.sequence
transform.sequence failures(propagate) {
^bb0(%op0: !transform.any_op, %val0: !transform.any_value, %par0: !transform.any_param):
  // CHECK: foreach %{{.*}} : !transform.any_op
  transform.foreach %op0 : !transform.any_op {
  ^bb1(%op1: !transform.any_op):
  }
  // CHECK: foreach %{{.*}} : !transform.any_op, !transform.any_value, !transform.any_param
  transform.foreach %op0, %val0, %par0 : !transform.any_op, !transform.any_value, !transform.any_param {
  ^bb1(%op1: !transform.any_op, %val1: !transform.any_value, %par1: !transform.any_param):
  }
  // CHECK: foreach %{{.*}} : !transform.any_op, !transform.any_value, !transform.any_param -> !transform.any_op
  transform.foreach %op0, %val0, %par0 : !transform.any_op, !transform.any_value, !transform.any_param -> !transform.any_op {
  ^bb1(%op1: !transform.any_op, %val1: !transform.any_value, %par1: !transform.any_param):
    transform.yield %op1 : !transform.any_op
  }
  // CHECK: foreach %{{.*}} : !transform.any_op, !transform.any_value, !transform.any_param -> !transform.any_param, !transform.any_value
  transform.foreach %op0, %val0, %par0 : !transform.any_op, !transform.any_value, !transform.any_param -> !transform.any_param, !transform.any_value {
  ^bb1(%op1: !transform.any_op, %val1: !transform.any_value, %par1: !transform.any_param):
    transform.yield %par1, %val1 : !transform.any_param, !transform.any_value
  }
}

// CHECK: transform.sequence
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  // CHECK: cast %{{.*}} : !transform.any_op to !transform.any_op
  %0 = cast %arg0: !transform.any_op to !transform.any_op
  // CHECK: cast %{{.*}} : !transform.any_op to !transform.op<"builtin.module">
  %1 = cast %0: !transform.any_op to !transform.op<"builtin.module">
}

// CHECK: transform.sequence
// CHECK-COUNT-9: print
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  transform.print %arg0 : !transform.any_op
  transform.print
  transform.print %arg0 {name = "test"} : !transform.any_op
  transform.print {name = "test"}
  transform.print {name = "test", assume_verified}
  transform.print %arg0 {assume_verified} : !transform.any_op
  transform.print %arg0 {use_local_scope} : !transform.any_op
  transform.print %arg0 {skip_regions} : !transform.any_op
  transform.print %arg0 {assume_verified, use_local_scope, skip_regions} : !transform.any_op
}

// CHECK: transform.sequence
// CHECK: transform.structured.tile_using_for %0 tile_sizes [4, 4, [4]]
transform.sequence failures(propagate) {
^bb0(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  transform.structured.tile_using_for %0 tile_sizes [4, 4, [4]] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
}

// CHECK: transform.sequence
// CHECK: transform.structured.tile_using_for %0 tile_sizes {{\[}}[2], 4, 8]
transform.sequence failures(propagate) {
^bb0(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  transform.structured.tile_using_for %0 tile_sizes [[2], 4, 8] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
}

// CHECK: transform.sequence
// CHECK: transform.param.constant "example_string
transform.sequence failures(propagate) {
^bb0(%arg1: !transform.any_op):
  transform.param.constant "example_string" -> !transform.any_param
}
