// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Check that dialect attributes are processed for external functions.
// This might not be an intended use case for `nvvm.minctasm`, but it enables
// testing this feature easily.

module {
  llvm.func external @f() attributes { nvvm.minctasm = 10 : i32 }
  // CHECK: !nvvm.annotations = !{![[NVVM:[0-9]+]]}
  // CHECK: ![[NVVM]] = !{ptr @f, !"minctasm", i32 10}
}
