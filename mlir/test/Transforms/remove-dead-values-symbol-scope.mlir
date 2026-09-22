// RUN: mlir-opt %s -split-input-file --pass-pipeline='builtin.module(gpu.module(remove-dead-values{canonicalize=false}),builtin.module(remove-dead-values{canonicalize=false}))' | FileCheck %s --check-prefixes=CHECK,SCOPED
// RUN: mlir-opt %s -split-input-file --pass-pipeline='builtin.module(gpu.module(remove-dead-values),builtin.module(remove-dead-values))' | FileCheck %s --check-prefixes=CHECK,SCOPED
// RUN: mlir-opt %s -split-input-file --remove-dead-values='canonicalize=false' | FileCheck %s --check-prefixes=CHECK,FULL

// The launch is outside the pass root. An empty user map does not mean that the
// kernel has no users. Keep its signature, including the unused argument.
module attributes {gpu.container_module} {
  gpu.module @kernels {
    // CHECK-LABEL: gpu.func @kernel(
    // CHECK-SAME: %{{[^ ,)]+}}: i32, %{{[^ ,)]+}}: i32) kernel
    gpu.func @kernel(%used: i32, %dead: i32) kernel attributes {sym_visibility = "nested"} {
      gpu.printf "value: %d", %used : i32
      gpu.return
    }
  }
  func.func @launch(%x: i32) {
    %c1 = arith.constant 1 : index
    // CHECK: gpu.launch_func @kernels::@kernel
    // CHECK-SAME: args(%{{[^ ,)]+}} : i32, %{{[^ ,)]+}} : i32)
    gpu.launch_func @kernels::@kernel blocks in (%c1, %c1, %c1)
        threads in (%c1, %c1, %c1) args(%x : i32, %x : i32)
    return
  }
}

// -----

module @outer {
  module @exposed {
    // A local call does not make the user map complete. Keep both arguments and
    // results when the outer call is outside the pass root.
    // SCOPED-LABEL: func.func nested @callee(
    // SCOPED-SAME: %{{[^ ,)]+}}: i32, %{{[^ ,)]+}}: i32) -> (i32, i32)
    // FULL-LABEL: func.func nested @callee(
    // FULL-SAME: %{{[^ ,)]+}}: i32) -> i32
    func.func nested @callee(%used: i32, %dead: i32) -> (i32, i32) {
      return %used, %used : i32, i32
    }
    func.func @local(%x: i32) {
      // SCOPED: call @callee(%{{[^ ,)]+}}, %{{[^ ,)]+}}) : (i32, i32) -> (i32, i32)
      // FULL: call @callee(%{{[^ ,)]+}}) : (i32) -> i32
      %r:2 = func.call @callee(%x, %x) : (i32, i32) -> (i32, i32)
      return
    }

    // Visibility must propagate through all exposed tables below the root.
    module @child attributes {sym_visibility = "nested"} {
      // SCOPED-LABEL: func.func nested @deep(
      // SCOPED-SAME: %{{[^ ,)]+}}: i32)
      // FULL-LABEL: func.func nested @deep()
      func.func nested @deep(%dead: i32) {
        return
      }
    }

    // Public functions can have callers outside the IR, even if all IR users
    // are visible to the pass.
    // CHECK-LABEL: func.func @public_callee(
    // CHECK-SAME: %{{[^ ,)]+}}: i32) -> i32
    func.func @public_callee(%dead: i32) -> i32 {
      %c1 = arith.constant 1 : i32
      return %c1 : i32
    }
    func.func @public_caller(%x: i32) {
      // CHECK: call @public_callee(%{{[^ ,)]+}}) : (i32) -> i32
      %r = func.call @public_callee(%x) : (i32) -> i32
      return
    }

    // Private symbols have no callers outside their table.
    // CHECK-LABEL: func.func private @private_callee()
    func.func private @private_callee(%dead: i32) {
      return
    }
    func.func @private_caller(%x: i32) {
      // CHECK: call @private_callee() : () -> ()
      func.call @private_callee(%x) : (i32) -> ()
      return
    }

    // A private table hides the nested symbols below it.
    module @hidden attributes {sym_visibility = "private"} {
      // CHECK-LABEL: func.func nested @hidden_callee()
      // CHECK-NEXT: return
      func.func nested @hidden_callee(%dead: i32) -> i32 {
        %c1 = arith.constant 1 : i32
        return %c1 : i32
      }
      func.func @hidden_caller(%x: i32) {
        // CHECK: call @hidden_callee() : () -> ()
        %r = func.call @hidden_callee(%x) : (i32) -> i32
        return
      }
    }
  }
  func.func @outside(%x: i32) -> i32 {
    // SCOPED: "test.conversion_call_op"(%{{[^ ,)]+}}, %{{[^ ,)]+}})
    // SCOPED-SAME: callee = @exposed::@callee
    // SCOPED-SAME: (i32, i32) -> (i32, i32)
    // FULL: "test.conversion_call_op"(%{{[^ ,)]+}})
    // FULL-SAME: callee = @exposed::@callee
    // FULL-SAME: (i32) -> i32
    %r:2 = "test.conversion_call_op"(%x, %x) {callee = @exposed::@callee}
        : (i32, i32) -> (i32, i32)
    // SCOPED: "test.conversion_call_op"(%{{[^ ,)]+}})
    // SCOPED-SAME: callee = @exposed::@child::@deep
    // SCOPED-SAME: (i32) -> ()
    // FULL: "test.conversion_call_op"()
    // FULL-SAME: callee = @exposed::@child::@deep
    // FULL-SAME: () -> ()
    "test.conversion_call_op"(%x) {callee = @exposed::@child::@deep} : (i32) -> ()
    return %r#0 : i32
  }
}
