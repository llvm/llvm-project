// RUN: mlir-opt %s -split-input-file -allow-unregistered-dialect --pass-pipeline='builtin.module(gpu.module(remove-dead-values{canonicalize=false}),builtin.module(remove-dead-values{canonicalize=false}))' | FileCheck %s
// RUN: mlir-opt %s -split-input-file -allow-unregistered-dialect --pass-pipeline='builtin.module(gpu.module(remove-dead-values),builtin.module(remove-dead-values))' | FileCheck %s
// RUN: mlir-opt %s -split-input-file -allow-unregistered-dialect --remove-dead-values='canonicalize=false' | FileCheck %s

// An unregistered op with a region may reference any symbol visible from its
// symbol table, so SymbolUserMap cannot compute all of its uses. It must not
// crash the pass, and symbols it may use must be treated conservatively.
// See https://github.com/llvm/llvm-project/issues/226052.
//
// Every case is wrapped in an outer module so that the nested pipelines above
// run the pass on the inner module.

// Original reproducer. The value consumed by the unknown op must stay alive.
// CHECK-LABEL: func.func @bbarg_of_unknown_op_2(
// CHECK-SAME: %{{[^ ,)]+}}: f32)
// CHECK: tensor.empty
// CHECK: linalg.fill
// CHECK: "dummy.dummy_op"
module {
  module {
    func.func @bbarg_of_unknown_op_2(%arg0: f32) {
      %0 = tensor.empty() : tensor<10xf32>
      %1 = linalg.fill ins(%arg0 : f32) outs(%0 : tensor<10xf32>) -> tensor<10xf32>
      "dummy.dummy_op"(%1) ({
      }) : (tensor<10xf32>) -> ()
    }
  }
}

// -----

// The unknown op is a direct child of the symbol table, next to the function.
// CHECK-LABEL: func.func @unknown_op_at_module_level(
// CHECK-SAME: %{{[^ ,)]+}}: f32) -> tensor<10xf32>
// CHECK: tensor.empty
// CHECK: linalg.fill
// CHECK: "dummy.top_level_op"
module {
  module {
    func.func @unknown_op_at_module_level(%arg0: f32) -> tensor<10xf32> {
      %0 = tensor.empty() : tensor<10xf32>
      %1 = linalg.fill ins(%arg0 : f32) outs(%0 : tensor<10xf32>) -> tensor<10xf32>
      return %1 : tensor<10xf32>
    }
    "dummy.top_level_op"() ({
    }) : () -> ()
  }
}

// -----

// The unknown op has a region with a block argument and a body.
// CHECK-LABEL: func.func @unknown_op_with_block_arg(
// CHECK-SAME: %{{[^ ,)]+}}: f32)
// CHECK: "dummy.dummy_op"
// CHECK-NEXT: ^bb0(%{{[^ ,)]+}}: tensor<10xf32>):
// CHECK-NEXT: "dummy.yield"
module {
  module {
    func.func @unknown_op_with_block_arg(%arg0: f32) {
      %0 = tensor.empty() : tensor<10xf32>
      %1 = linalg.fill ins(%arg0 : f32) outs(%0 : tensor<10xf32>) -> tensor<10xf32>
      "dummy.dummy_op"(%1) ({
      ^bb0(%arg1: tensor<10xf32>):
        "dummy.yield"(%arg1) : (tensor<10xf32>) -> ()
      }) : (tensor<10xf32>) -> ()
      return
    }
  }
}

// -----

// Dead values unrelated to the unknown op are still removed.
// CHECK-LABEL: func.func @dead_value_next_to_unknown_op(
// CHECK-SAME: %{{[^ ,)]+}}: f32)
// CHECK-NOT: tensor.empty
// CHECK-NOT: linalg.fill
// CHECK: "dummy.dummy_op"
module {
  module {
    func.func @dead_value_next_to_unknown_op(%arg0: f32) {
      %0 = tensor.empty() : tensor<10xf32>
      %1 = linalg.fill ins(%arg0 : f32) outs(%0 : tensor<10xf32>) -> tensor<10xf32>
      "dummy.dummy_op"() ({
      }) : () -> ()
      return
    }
  }
}

// -----

// Control, no unknown op: the dead argument of a private callee is removed.
// CHECK-LABEL: func.func private @callee_control(
// CHECK-SAME: %{{[^ ,)]+}}: i32) -> i32
// CHECK-LABEL: func.func @caller_control(
// CHECK: call @callee_control(%{{[^ ,)]+}}) : (i32) -> i32
module {
  module {
    func.func private @callee_control(%used: i32, %dead: i32) -> i32 {
      return %used : i32
    }
    func.func @caller_control(%x: i32) -> i32 {
      %r = func.call @callee_control(%x, %x) : (i32, i32) -> i32
      return %r : i32
    }
  }
}

// -----

// Same as the control plus an unknown op with a region. Not all users of the
// callee are known, so keep its signature, including the unused argument.
// CHECK-LABEL: func.func private @callee_unknown_user(
// CHECK-SAME: %{{[^ ,)]+}}: i32, %{{[^ ,)]+}}: i32) -> i32
// CHECK-LABEL: func.func @caller_unknown_user(
// CHECK: call @callee_unknown_user(%{{[^ ,)]+}}, %{{[^ ,)]+}}) : (i32, i32) -> i32
// CHECK: "dummy.dummy_op"
module {
  module {
    func.func private @callee_unknown_user(%used: i32, %dead: i32) -> i32 {
      return %used : i32
    }
    func.func @caller_unknown_user(%x: i32) -> i32 {
      %r = func.call @callee_unknown_user(%x, %x) : (i32, i32) -> i32
      "dummy.dummy_op"() ({
      }) : () -> ()
      return %r : i32
    }
  }
}

// -----

// The unknown op lives in a symbol table enclosing the callee, and that table
// is inside the pass root. The enclosing table must be taken into account when
// deciding that all uses are visible.
// CHECK-LABEL: module @outer
// CHECK: func.func private @callee_nested(
// CHECK-SAME: %{{[^ ,)]+}}: i32, %{{[^ ,)]+}}: i32) -> i32
// CHECK: call @callee_nested(%{{[^ ,)]+}}, %{{[^ ,)]+}}) : (i32, i32) -> i32
// CHECK: "dummy.dummy_op"
module {
  module @outer {
    module @inner {
      func.func private @callee_nested(%used: i32, %dead: i32) -> i32 {
        return %used : i32
      }
      func.func @caller_nested(%x: i32) -> i32 {
        %r = func.call @callee_nested(%x, %x) : (i32, i32) -> i32
        return %r : i32
      }
    }
    "dummy.dummy_op"() ({
    }) : () -> ()
  }
}
