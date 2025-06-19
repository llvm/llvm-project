// Test that attempting to create a local crash reproducer without disabling threading
// prints an error from the pass manager (as opposed to crashing with a stack trace).

// RUN: mlir-opt --mlir-pass-pipeline-local-reproducer --mlir-pass-pipeline-crash-reproducer=%t 2>&1 %s | FileCheck %s
// CHECK: error: Local crash reproduction may not be used without disabling mutli-threading first.
